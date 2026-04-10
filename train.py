"""
SPADE training script – terrain → water attribute (height / vel / foam)
Usage:
    python train.py --target height
    python train.py --target vel   --epochs 200 --batch 4
    python train.py --target foam  --epochs 200
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import HeightMapDataset, TARGET_CONFIGS
from models import SPADEGenerator, MultiscaleDiscriminator

ROOT = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


# ----------------------------- losses ---------------------------------

def hinge_loss_d(real_feats_list, fake_feats_list):
    loss = 0
    for real_feats, fake_feats in zip(real_feats_list, fake_feats_list):
        loss += torch.mean(F.relu(1.0 - real_feats[-1]))
        loss += torch.mean(F.relu(1.0 + fake_feats[-1]))
    return loss * 0.5


def hinge_loss_g(fake_feats_list):
    loss = 0
    for fake_feats in fake_feats_list:
        loss -= torch.mean(fake_feats[-1])
    return loss


def feat_matching_loss(real_feats_list, fake_feats_list):
    loss = 0.0
    count = 0
    for real_feats, fake_feats in zip(real_feats_list, fake_feats_list):
        for rf, ff in zip(real_feats[:-1], fake_feats[:-1]):
            loss += F.l1_loss(ff, rf.detach())
            count += 1
    return loss / max(count, 1)


def weighted_l1_loss(pred, target, label, fg_weight=10.0, threshold=0.005):
    """L1 with higher weight on terrain-foreground pixels (mask from label)."""
    mask = (label.sum(dim=1, keepdim=True) > threshold).float()
    weight = 1.0 + mask * (fg_weight - 1.0)
    return (torch.abs(pred - target) * weight).mean()


def masked_l1(pred, target, label, threshold=0.005):
    """L1 only on terrain-foreground pixels (metric)."""
    mask = (label.sum(dim=1, keepdim=True) > threshold).float()
    num = (torch.abs(pred - target) * mask).sum()
    denom = mask.sum().clamp(min=1.0)
    return num / denom


class VGGFeatureLoss(nn.Module):
    """Perceptual loss (adapted for 1 or 2 channel input)."""

    def __init__(self, in_nc=1):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18]])
        for p in self.parameters():
            p.requires_grad = False
        self.expand = nn.Conv2d(in_nc, 3, 1, bias=False)
        nn.init.constant_(self.expand.weight, 1.0 / 3.0)
        self.expand.weight.requires_grad = False
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8]

    def forward(self, pred, target):
        x, y = self.expand(pred), self.expand(target)
        loss = 0.0
        for s, w in zip(self.slices, self.weights):
            x, y = s(x), s(y)
            loss += w * F.l1_loss(x, y)
        return loss


# ----------------------------- scheduler ------------------------------

def build_scheduler(optimizer, total_epochs, decay_start):
    def _rule(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_rule)


# ----------------------------- preview --------------------------------

def _auto_contrast(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def save_preview(label, target, pred, path, target_type):
    """Save terrain | gt | pred side-by-side with auto-contrast."""
    lbl = _auto_contrast(label[0, 0].cpu().float().numpy())
    H = lbl.shape[0]
    gap = np.full((H, 4), 128, dtype=np.uint8)

    nc = target.shape[1]
    panels = [lbl, gap]

    for tensor in [target, pred]:
        if nc == 1:
            panels.append(_auto_contrast(tensor[0, 0].cpu().float().numpy()))
        else:
            for ch in range(nc):
                panels.append(_auto_contrast(tensor[0, ch].cpu().float().numpy()))
                if ch < nc - 1:
                    panels.append(np.full((H, 2), 64, dtype=np.uint8))
        panels.append(gap)

    combined = np.concatenate(panels[:-1], axis=1)
    Image.fromarray(combined, mode="L").save(path)


# ------------------------------ main ----------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="height",
                        choices=["height", "vel", "foam"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)
    parser.add_argument("--lambda_l1", type=float, default=20.0)
    parser.add_argument("--lambda_vgg", type=float, default=5.0)
    parser.add_argument("--lambda_fm", type=float, default=10.0)
    parser.add_argument("--fg_weight", type=float, default=10.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--decay_epoch", type=int, default=100)
    args = parser.parse_args()

    cfg = TARGET_CONFIGS[args.target]
    output_nc = cfg["channels"]
    fg_thresh = cfg["fg_threshold"]
    use_vgg = args.target != "vel"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | Target: {args.target} | OutputCh: {output_nc}")

    # ---- data --------------------------------------------------------
    train_ds = HeightMapDataset(ROOT, target_type=args.target,
                                split="train", augment=True, img_size=args.img_size)
    val_ds = HeightMapDataset(ROOT, target_type=args.target,
                              split="val", augment=False, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True)
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, ImgSize: {args.img_size}")

    # ---- models ------------------------------------------------------
    netG = SPADEGenerator(label_nc=1, output_nc=output_nc, ngf=args.ngf).to(device)
    netD = MultiscaleDiscriminator(input_nc=1 + output_nc).to(device)

    vgg_loss_fn = None
    if use_vgg:
        vgg_loss_fn = VGGFeatureLoss(in_nc=output_nc).to(device)

    total_g = sum(p.numel() for p in netG.parameters()) / 1e6
    total_d = sum(p.numel() for p in netD.parameters()) / 1e6
    log(f"G: {total_g:.1f}M params, D: {total_d:.1f}M params")

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.0, 0.999))
    schedG = build_scheduler(optG, args.epochs, args.decay_epoch)
    schedD = build_scheduler(optD, args.epochs, args.decay_epoch)

    # ---- resume ------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch):
            schedG.step()
            schedD.step()
        log(f"Resumed from epoch {start_epoch}")

    # ---- dirs (per-target) -------------------------------------------
    ckpt_dir = os.path.join(ROOT, "checkpoints", args.target)
    preview_dir = os.path.join(ROOT, "results_preview", args.target)
    run_dir = os.path.join(ROOT, "runs", args.target)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)

    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_masked = float("inf")

    # ---- training loop -----------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        netG.train()
        netD.train()
        t0 = time.time()
        sum_gL, sum_dL, n_steps = 0.0, 0.0, 0

        for step, (label, target) in enumerate(train_loader):
            label = label.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # ---------- D step ----------
            optD.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = netG(label).detach()
                d_real = netD(label, target)
                d_fake = netD(label, fake)
                loss_d = hinge_loss_d(d_real, d_fake)

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)
            scaler_d.step(optD)
            scaler_d.update()

            # ---------- G step ----------
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = netG(label)
                d_real = netD(label, target)
                d_fake = netD(label, fake)

                loss_gan = hinge_loss_g(d_fake)
                loss_fm = feat_matching_loss(d_real, d_fake) * args.lambda_fm
                loss_l1 = weighted_l1_loss(fake, target, label,
                                           fg_weight=args.fg_weight,
                                           threshold=fg_thresh) * args.lambda_l1
                loss_g = loss_gan + loss_fm + loss_l1

                if vgg_loss_fn is not None:
                    loss_g = loss_g + vgg_loss_fn(fake, target) * args.lambda_vgg

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.0)
            scaler_g.step(optG)
            scaler_g.update()

            sum_gL += loss_g.item()
            sum_dL += loss_d.item()
            n_steps += 1

            if step % 200 == 0 and step > 0:
                log(f"  step {step}/{len(train_loader)} "
                    f"G={loss_g.item():.4f} D={loss_d.item():.4f}")

        schedG.step()
        schedD.step()

        # ---------- validation ----------
        netG.eval()
        val_l1_sum, val_ml1_sum, val_n = 0.0, 0.0, 0
        sample_saved = False
        with torch.no_grad():
            for label, target in val_loader:
                label = label.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake = netG(label)
                bs = label.size(0)
                val_l1_sum += F.l1_loss(fake, target).item() * bs
                val_ml1_sum += masked_l1(fake, target, label, threshold=fg_thresh).item() * bs
                val_n += bs
                if not sample_saved:
                    save_preview(label, target, fake,
                                 os.path.join(preview_dir, f"epoch_{epoch + 1:04d}.png"),
                                 args.target)
                    sample_saved = True

        val_l1 = val_l1_sum / max(val_n, 1)
        val_masked = val_ml1_sum / max(val_n, 1)
        dt = time.time() - t0
        avg_g = sum_gL / max(n_steps, 1)
        avg_d = sum_dL / max(n_steps, 1)
        cur_lr = schedG.get_last_lr()[0]

        writer.add_scalar("Loss/G_total", avg_g, epoch)
        writer.add_scalar("Loss/D", avg_d, epoch)
        writer.add_scalar("Val/L1", val_l1, epoch)
        writer.add_scalar("Val/MaskedL1", val_masked, epoch)
        writer.add_scalar("LR/G", cur_lr, epoch)

        log(f"[{args.target}] [Epoch {epoch + 1}/{args.epochs}] G={avg_g:.4f} D={avg_d:.4f} "
            f"ValL1={val_l1:.5f} FgL1={val_masked:.5f} LR={cur_lr:.2e} ({dt:.1f}s)")

        # ---------- checkpoints ----------
        if val_masked < best_val_masked:
            best_val_masked = val_masked
            torch.save(netG.state_dict(), os.path.join(ckpt_dir, "best_G.pth"))
            log(f"  -> Best model saved (FgL1={val_masked:.5f})")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
            }, os.path.join(ckpt_dir, f"ckpt_epoch{epoch + 1}.pth"))

    writer.close()
    log(f"[{args.target}] Training complete.")


if __name__ == "__main__":
    main()
