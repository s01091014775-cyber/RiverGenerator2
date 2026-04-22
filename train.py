"""
SPADE training script – terrain → water attribute (height / vel / foam)
Usage:
    python train.py --target height
    python train.py --target vel   --epochs 200 --batch 4
    python train.py --target foam  --epochs 200
    python train.py --target vel   --pretrain --epochs 50   (phase 1: regression only)
    python train.py --target vel   --load_g checkpoints/vel/pretrain_G.pth  (phase 2: GAN fine-tune)
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


def spatial_gradient_loss(pred, target):
    """Penalise differences in spatial gradients (Sobel-like) between pred and target.
    Encourages spatially coherent vector fields — especially important for vel."""
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_tgt) + F.l1_loss(dy_pred, dy_tgt)


def angular_velocity_loss(pred, target, label, threshold=0.01):
    """Cosine-distance loss on 2-ch velocity vectors.
    Only computed on foreground pixels where velocity matters.
    Penalizes direction errors independent of magnitude."""
    assert pred.shape[1] == 2
    mask = (label.sum(dim=1, keepdim=True) > threshold).float()
    pred_centered = pred - 0.5
    tgt_centered = target - 0.5

    pred_mag = torch.sqrt((pred_centered ** 2).sum(dim=1, keepdim=True) + 1e-8)
    tgt_mag = torch.sqrt((tgt_centered ** 2).sum(dim=1, keepdim=True) + 1e-8)

    cos_sim = (pred_centered * tgt_centered).sum(dim=1, keepdim=True) / (pred_mag * tgt_mag)
    angular_err = (1.0 - cos_sim) * mask

    n_fg = mask.sum().clamp(min=1.0)
    return angular_err.sum() / n_fg


def magnitude_loss(pred, target, label, threshold=0.01):
    """L1 loss on velocity magnitude — ensures flow speed is preserved."""
    assert pred.shape[1] == 2
    mask = (label.sum(dim=1, keepdim=True) > threshold).float()
    pred_mag = torch.sqrt(((pred - 0.5) ** 2).sum(dim=1, keepdim=True) + 1e-8)
    tgt_mag = torch.sqrt(((target - 0.5) ** 2).sum(dim=1, keepdim=True) + 1e-8)
    diff = torch.abs(pred_mag - tgt_mag) * mask
    n_fg = mask.sum().clamp(min=1.0)
    return diff.sum() / n_fg


def r1_gradient_penalty(d_real_feats, real_images):
    """R1 gradient penalty to prevent D from becoming too strong."""
    grad = torch.autograd.grad(
        outputs=[f[-1].sum() for f in d_real_feats],
        inputs=real_images,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


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

def build_scheduler(optimizer, total_epochs, decay_start, mode="linear"):
    if mode == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
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
                        choices=["height", "vel", "vel_x25", "foam"])
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
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--lambda_grad", type=float, default=0.0,
                        help="Spatial gradient loss weight (recommended ~5 for vel)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable self-attention in generator bottleneck")
    parser.add_argument("--output_act", type=str, default="sigmoid",
                        choices=["sigmoid", "hardtanh"],
                        help="Generator output activation (hardtanh gives uniform gradient in [0,1])")
    parser.add_argument("--lambda_angular", type=float, default=0.0,
                        help="Angular velocity loss weight (recommended ~5-10 for vel)")
    parser.add_argument("--lambda_mag", type=float, default=0.0,
                        help="Magnitude loss weight (recommended ~5 for vel)")
    parser.add_argument("--pretrain", action="store_true",
                        help="Phase 1: train G only with regression losses (no GAN/D)")
    parser.add_argument("--load_g", type=str, default=None,
                        help="Load only G weights from file (for phase 2 after pretrain)")
    parser.add_argument("--r1_gamma", type=float, default=0.0,
                        help="R1 gradient penalty weight for D (recommended ~10)")
    parser.add_argument("--r1_every", type=int, default=16,
                        help="Apply R1 penalty every N D steps")
    args = parser.parse_args()

    cfg = TARGET_CONFIGS[args.target]
    output_nc = cfg["channels"]
    fg_thresh = cfg["fg_threshold"]
    use_vgg = True
    use_grad = args.lambda_grad > 0
    use_angular = args.lambda_angular > 0 and output_nc == 2
    use_mag = args.lambda_mag > 0 and output_nc == 2
    pretrain = args.pretrain
    use_r1 = args.r1_gamma > 0 and not pretrain

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "PRETRAIN (no GAN)" if pretrain else "GAN"
    log(f"Device: {device} | Target: {args.target} | OutputCh: {output_nc} | Mode: {mode_str}")

    # ---- data --------------------------------------------------------
    train_ds = HeightMapDataset(ROOT, target_type=args.target,
                                split="train", augment=True, img_size=args.img_size)
    val_ds = HeightMapDataset(ROOT, target_type=args.target,
                              split="val", augment=False, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True,
                            persistent_workers=True)
    log(f"Train: {len(train_ds)}, Val: {len(val_ds)}, ImgSize: {args.img_size}")

    # ---- models ------------------------------------------------------
    netG = SPADEGenerator(label_nc=1, output_nc=output_nc, ngf=args.ngf,
                          use_attention=not args.no_attention,
                          output_act=args.output_act).to(device)
    netD = MultiscaleDiscriminator(input_nc=1 + output_nc).to(device)

    vgg_loss_fn = None
    if use_vgg:
        vgg_loss_fn = VGGFeatureLoss(in_nc=output_nc).to(device)

    total_g = sum(p.numel() for p in netG.parameters()) / 1e6
    total_d = sum(p.numel() for p in netD.parameters()) / 1e6
    log(f"G: {total_g:.1f}M params, D: {total_d:.1f}M params")

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.0, 0.999))
    schedG = build_scheduler(optG, args.epochs, args.decay_epoch, args.scheduler)
    schedD = build_scheduler(optD, args.epochs, args.decay_epoch, args.scheduler)

    # ---- resume / load ------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = netG.load_state_dict(ckpt["netG"], strict=False)
        if missing:
            log(f"  G missing keys (will init fresh): {missing}")
        if unexpected:
            log(f"  G unexpected keys (ignored): {unexpected}")
        if not pretrain:
            netD.load_state_dict(ckpt["netD"], strict=False)
        try:
            optG.load_state_dict(ckpt["optG"])
        except (ValueError, KeyError):
            log("  optG state incompatible, reinitializing optimizer")
        if not pretrain:
            try:
                optD.load_state_dict(ckpt["optD"])
            except (ValueError, KeyError):
                log("  optD state incompatible, reinitializing optimizer")
        start_epoch = ckpt["epoch"] + 1
        if "schedG" in ckpt and "schedD" in ckpt:
            schedG.load_state_dict(ckpt["schedG"])
            if not pretrain:
                schedD.load_state_dict(ckpt["schedD"])
        else:
            for _ in range(start_epoch):
                schedG.step()
                if not pretrain:
                    schedD.step()
        if "best_val_masked" in ckpt:
            best_val_masked = ckpt["best_val_masked"]
        log(f"Resumed from epoch {start_epoch}")
    elif args.load_g and os.path.isfile(args.load_g):
        state = torch.load(args.load_g, map_location=device, weights_only=False)
        if "netG" in state:
            state = state["netG"]
        missing, unexpected = netG.load_state_dict(state, strict=False)
        if missing:
            log(f"  G missing keys: {missing}")
        if unexpected:
            log(f"  G unexpected keys: {unexpected}")
        log(f"Loaded G weights from {args.load_g} (fresh D & optimizers)")

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
    patience_counter = 0

    global_step = 0

    # ---- training loop -----------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        netG.train()
        if not pretrain:
            netD.train()
        t0 = time.time()
        sum_gL, sum_dL, sum_gan, sum_fm, sum_l1_tr, n_steps = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for step, (label, target) in enumerate(train_loader):
            label = label.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if not pretrain:
                # ---------- D step ----------
                optD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        fake = netG(label)

                if use_r1 and global_step % args.r1_every == 0:
                    target.requires_grad_(True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    d_real = netD(label, target)
                    d_fake = netD(label, fake.detach())
                    loss_d = hinge_loss_d(d_real, d_fake)

                if use_r1 and global_step % args.r1_every == 0:
                    r1_pen = r1_gradient_penalty(d_real, target)
                    loss_d = loss_d + args.r1_gamma * 0.5 * r1_pen * args.r1_every
                    target.requires_grad_(False)

                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(optD)
                torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)
                scaler_d.step(optD)
                scaler_d.update()

            # ---------- G step ----------
            optG.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = netG(label)

                loss_l1 = weighted_l1_loss(fake, target, label,
                                           fg_weight=args.fg_weight,
                                           threshold=fg_thresh) * args.lambda_l1
                loss_g = loss_l1

                if vgg_loss_fn is not None:
                    loss_g = loss_g + vgg_loss_fn(fake, target) * args.lambda_vgg
                if use_grad:
                    loss_g = loss_g + spatial_gradient_loss(fake, target) * args.lambda_grad
                if use_angular:
                    loss_g = loss_g + angular_velocity_loss(fake, target, label, fg_thresh) * args.lambda_angular
                if use_mag:
                    loss_g = loss_g + magnitude_loss(fake, target, label, fg_thresh) * args.lambda_mag

                loss_gan_val = 0.0
                loss_fm_val = 0.0
                if not pretrain:
                    d_real = netD(label, target)
                    d_fake = netD(label, fake)
                    loss_gan = hinge_loss_g(d_fake)
                    loss_fm = feat_matching_loss(d_real, d_fake) * args.lambda_fm
                    loss_g = loss_g + loss_gan + loss_fm
                    loss_gan_val = loss_gan.item()
                    loss_fm_val = loss_fm.item()

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.0)
            scaler_g.step(optG)
            scaler_g.update()

            g_val = loss_g.item()
            d_val = loss_d.item() if not pretrain else 0.0
            if not (np.isnan(g_val) or np.isnan(d_val)):
                sum_gL += g_val
                sum_dL += d_val
                sum_gan += loss_gan_val
                sum_fm += loss_fm_val
                sum_l1_tr += loss_l1.item()
                n_steps += 1

            global_step += 1

            if step % 200 == 0 and step > 0:
                if pretrain:
                    log(f"  step {step}/{len(train_loader)} G={loss_g.item():.4f}")
                else:
                    log(f"  step {step}/{len(train_loader)} "
                        f"G={loss_g.item():.4f} D={d_val:.4f}")

        schedG.step()
        if not pretrain:
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
        writer.add_scalar("Loss/G_l1", sum_l1_tr / max(n_steps, 1), epoch)
        writer.add_scalar("Val/L1", val_l1, epoch)
        writer.add_scalar("Val/MaskedL1", val_masked, epoch)
        writer.add_scalar("LR/G", cur_lr, epoch)
        if not pretrain:
            writer.add_scalar("Loss/D", avg_d, epoch)
            writer.add_scalar("Loss/G_gan", sum_gan / max(n_steps, 1), epoch)
            writer.add_scalar("Loss/G_fm", sum_fm / max(n_steps, 1), epoch)

        if pretrain:
            log(f"[{args.target}] [Epoch {epoch + 1}/{args.epochs}] G={avg_g:.4f} "
                f"ValL1={val_l1:.5f} FgL1={val_masked:.5f} LR={cur_lr:.2e} ({dt:.1f}s)")
        else:
            log(f"[{args.target}] [Epoch {epoch + 1}/{args.epochs}] G={avg_g:.4f} D={avg_d:.4f} "
                f"ValL1={val_l1:.5f} FgL1={val_masked:.5f} LR={cur_lr:.2e} ({dt:.1f}s)")

        # ---------- checkpoints ----------
        best_name = "pretrain_G.pth" if pretrain else "best_G.pth"
        if val_masked < best_val_masked:
            best_val_masked = val_masked
            patience_counter = 0
            torch.save(netG.state_dict(), os.path.join(ckpt_dir, best_name))
            log(f"  -> Best model saved as {best_name} (FgL1={val_masked:.5f})")
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            ckpt_data = {
                "epoch": epoch,
                "netG": netG.state_dict(),
                "optG": optG.state_dict(),
                "schedG": schedG.state_dict(),
                "best_val_masked": best_val_masked,
            }
            if not pretrain:
                ckpt_data["netD"] = netD.state_dict()
                ckpt_data["optD"] = optD.state_dict()
                ckpt_data["schedD"] = schedD.state_dict()
            torch.save(ckpt_data, os.path.join(ckpt_dir, f"ckpt_epoch{epoch + 1}.pth"))

        if args.patience > 0 and patience_counter >= args.patience:
            log(f"  Early stopping triggered (no improvement for {args.patience} epochs)")
            break

    writer.close()
    log(f"[{args.target}] Training complete ({mode_str}). Best FgL1={best_val_masked:.5f}")


if __name__ == "__main__":
    main()
