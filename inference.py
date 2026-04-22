"""
Inference & evaluation – load trained SPADE generator and predict water attributes.
Usage:
    python inference.py --target height --eval --limit 30
    python inference.py --target vel    --input terrain.png --output vel_out.png
    python inference.py --target foam   --input_dir datasets/train_label --output_dir results/foam
"""

import argparse
import os
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HeightMapDataset, TARGET_CONFIGS, TERRAIN_MAX, VEL_BG, VEL_SCALE

def _get_vel_amplify(target_type):
    cfg = TARGET_CONFIGS.get(target_type, {})
    return cfg.get("amplify", 1.0)
from models import SPADEGenerator

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_terrain(path, img_size=256):
    arr = np.array(Image.open(path), dtype=np.float32) / TERRAIN_MAX
    arr = np.clip(arr, 0.0, 1.0)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    if t.shape[2] != img_size or t.shape[3] != img_size:
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t


def save_output(tensor, path, target_type, target_size=512):
    """Save prediction back in the original data format."""
    t = tensor
    if t.shape[-1] != target_size:
        t = F.interpolate(t.unsqueeze(0) if t.dim() == 3 else t,
                          size=(target_size, target_size),
                          mode="bilinear", align_corners=False)
        if t.dim() == 4:
            t = t.squeeze(0)
    arr = t.cpu().numpy()

    if target_type == "height":
        out = np.clip(arr[0] * 500.0, 0, 65535).astype(np.uint16)
        Image.fromarray(out, mode="I;16").save(path)
    elif target_type in ("vel", "vel_x25"):
        amp = _get_vel_amplify(target_type)
        h, w = arr.shape[1], arr.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        r_px = (arr[0] - 0.5) / amp * VEL_SCALE + VEL_BG
        g_px = (arr[1] - 0.5) / amp * VEL_SCALE + VEL_BG
        rgb[:, :, 0] = np.clip(r_px, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip(g_px, 0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path)
    elif target_type == "foam":
        out = np.clip(arr[0] * 255, 0, 255).astype(np.uint8)
        Image.fromarray(out, mode="L").save(path)


def _auto_contrast(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def _make_vel_rgb(tensor, amplify=1.0):
    """Combine 2-ch velocity tensor (R=vx, G=vy) into an RGB image.
    Reverses normalization; amplify>1 means values were amplified during training."""
    arr = tensor.cpu().numpy()
    h, w = arr.shape[1], arr.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    r_px = (arr[0] - 0.5) / amplify * VEL_SCALE + VEL_BG
    g_px = (arr[1] - 0.5) / amplify * VEL_SCALE + VEL_BG
    rgb[:, :, 0] = np.clip(r_px, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip(g_px, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = 0
    return rgb


def save_comparison(label, target, pred, path, target_type):
    lbl = _auto_contrast(label.squeeze().cpu().numpy())
    H = lbl.shape[0]
    nc = target.shape[0]
    amp = _get_vel_amplify(target_type)

    if nc == 2:
        lbl_rgb = np.stack([lbl, lbl, lbl], axis=-1)
        gap = np.full((H, 4, 3), 128, dtype=np.uint8)
        gt_rgb = _make_vel_rgb(target, amp)
        pred_rgb = _make_vel_rgb(pred, amp)
        combined = np.concatenate([lbl_rgb, gap, gt_rgb, gap, pred_rgb], axis=1)
        Image.fromarray(combined, mode="RGB").save(path)
    else:
        gap = np.full((H, 4), 128, dtype=np.uint8)
        panels = [lbl, gap]
        for tensor in [target, pred]:
            panels.append(_auto_contrast(tensor[0].cpu().numpy()))
            panels.append(gap)
        combined = np.concatenate(panels[:-1], axis=1)
        Image.fromarray(combined, mode="L").save(path)


def compute_metrics(pred, target, label, threshold=0.005):
    diff = torch.abs(pred - target)
    l1_full = diff.mean().item()

    mask = (label.sum(dim=0, keepdim=True) > threshold).float()
    n_fg = mask.sum().item()
    l1_fg = (diff * mask).sum().item() / max(n_fg, 1.0)

    bg_mask = 1.0 - mask
    n_bg = bg_mask.sum().item()
    l1_bg = (diff * bg_mask).sum().item() / max(n_bg, 1.0)

    mse = (diff ** 2).mean().item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))

    return {"l1": l1_full, "l1_fg": l1_fg, "l1_bg": l1_bg, "psnr": psnr}


def run_eval(args):
    cfg = TARGET_CONFIGS[args.target]
    output_nc = cfg["channels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = SPADEGenerator(label_nc=1, output_nc=output_nc, ngf=args.ngf,
                          use_attention=not args.no_attention,
                          output_act=args.output_act).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    netG.load_state_dict(state)
    netG.eval()
    print(f"Loaded: {args.checkpoint}")

    val_ds = HeightMapDataset(ROOT, target_type=args.target,
                              split="val", augment=False, img_size=args.img_size)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Val samples: {len(val_ds)}")

    out_dir = os.path.join(ROOT, "test_results", args.target)
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = []
    count = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for i, (label, target) in enumerate(val_loader):
            if count >= args.limit:
                break
            label = label.to(device)
            target = target.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = netG(label)
            m = compute_metrics(pred[0], target[0], label[0], threshold=cfg["fg_threshold"])
            all_metrics.append(m)

            save_comparison(label[0], target[0], pred[0],
                            os.path.join(out_dir, f"val_{i:04d}_cmp.png"),
                            args.target)
            save_output(pred[0], os.path.join(out_dir, f"val_{i:04d}_pred.png"),
                        args.target, target_size=args.out_size)
            count += 1

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"\n{'='*50}")
    print(f"  [{args.target}] Evaluation on {count} val samples")
    print(f"  Full L1  : {avg['l1']:.5f}")
    print(f"  FG L1    : {avg['l1_fg']:.5f}")
    print(f"  BG L1    : {avg['l1_bg']:.6f}")
    print(f"  PSNR     : {avg['psnr']:.2f} dB")
    print(f"{'='*50}")
    print(f"Results saved to: {out_dir}")

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Target: {args.target}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Samples: {count}\n")
        for k, v in avg.items():
            f.write(f"{k}: {v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="height",
                        choices=["height", "vel", "vel_x25", "foam"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--out_size", type=int, default=512)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable self-attention in generator bottleneck")
    parser.add_argument("--output_act", type=str, default="sigmoid",
                        choices=["sigmoid", "hardtanh"])
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(ROOT, "checkpoints", args.target, "best_G.pth")
    if args.output is None:
        args.output = f"{args.target}_out.png"
    if args.output_dir is None:
        args.output_dir = os.path.join("results", args.target)

    if args.eval:
        run_eval(args)
        return

    cfg = TARGET_CONFIGS[args.target]
    output_nc = cfg["channels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = SPADEGenerator(label_nc=1, output_nc=output_nc, ngf=args.ngf,
                          use_attention=not args.no_attention,
                          output_act=args.output_act).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    netG.load_state_dict(state)
    netG.eval()
    print(f"Loaded: {args.checkpoint}")

    if args.input:
        label = load_terrain(args.input, args.img_size).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = netG(label)
        save_output(pred[0], args.output, args.target, target_size=args.out_size)
        print(f"Saved -> {args.output}")

    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))[:args.limit]
        for p in paths:
            label = load_terrain(p, args.img_size).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                pred = netG(label)
            out_name = os.path.basename(p)
            save_output(pred[0], os.path.join(args.output_dir, out_name),
                        args.target, target_size=args.out_size)
        print(f"Saved {len(paths)} results to {args.output_dir}")
    else:
        print("Provide --input, --input_dir, or --eval")


if __name__ == "__main__":
    main()
