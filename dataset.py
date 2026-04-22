import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


VEL_BG = 186.0
VEL_SCALE = 186.0

TARGET_CONFIGS = {
    "height": {
        "subdir": "train_img_height",
        "channels": 1,
        "dtype": "uint16",
        "max_val": 500.0,
        "fg_threshold": 0.005,
    },
    "vel": {
        "subdir": "train_img_vel",
        "channels": 2,
        "dtype": "rgba_rg",
        "max_val": 255.0,
        "fg_threshold": 0.01,
        "bg_value": VEL_BG,
        "scale": VEL_SCALE,
    },
    "vel_x25": {
        "subdir": "train_img_vel_x25",
        "channels": 2,
        "dtype": "rgba_rg",
        "max_val": 255.0,
        "fg_threshold": 0.01,
        "bg_value": VEL_BG,
        "scale": VEL_SCALE,
        "amplify": 2.5,
    },
    "foam": {
        "subdir": "train_img_vel",
        "channels": 1,
        "dtype": "rgba_a",
        "max_val": 255.0,
        "fg_threshold": 0.01,
    },
}

TERRAIN_MAX = 2000.0


class HeightMapDataset(Dataset):
    """
    Paired terrain-label → target dataset.
    Supports multiple target types via `target_type`:
      - "height": uint16 water depth
      - "vel":    R/G channels from RGBA velocity image (2-ch)
      - "foam":   A channel from RGBA velocity image (1-ch)
    """

    def __init__(self, root_dir, target_type="height", split="train",
                 val_ratio=0.1, augment=False, img_size=256):
        assert target_type in TARGET_CONFIGS, f"Unknown target: {target_type}"
        self.cfg = TARGET_CONFIGS[target_type]
        self.target_type = target_type
        self.img_size = img_size
        self.augment = augment

        label_dir = os.path.join(root_dir, "datasets", "train_label")
        img_dir = os.path.join(root_dir, "datasets", self.cfg["subdir"])

        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        assert len(self.label_paths) == len(self.img_paths), \
            f"Mismatch: {len(self.label_paths)} labels vs {len(self.img_paths)} images"

        n = len(self.label_paths)
        n_val = int(n * val_ratio)
        if split == "train":
            self.label_paths = self.label_paths[n_val:]
            self.img_paths = self.img_paths[n_val:]
        else:
            self.label_paths = self.label_paths[:n_val]
            self.img_paths = self.img_paths[:n_val]

    def __len__(self):
        return len(self.label_paths)

    def _load_label(self, path):
        arr = np.array(Image.open(path), dtype=np.float32) / TERRAIN_MAX
        arr = np.clip(arr, 0.0, 1.0)
        t = torch.from_numpy(arr).unsqueeze(0)
        if t.shape[1] != self.img_size or t.shape[2] != self.img_size:
            t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        return t

    def _load_target(self, path):
        img = Image.open(path)
        dt = self.cfg["dtype"]

        if dt == "uint16":
            arr = np.array(img, dtype=np.float32) / self.cfg["max_val"]
            arr = np.clip(arr, 0.0, 1.0)
            t = torch.from_numpy(arr).unsqueeze(0)
        elif dt == "rgba_rg":
            arr = np.array(img, dtype=np.float32)
            rg = arr[:, :, :2]
            bg = self.cfg["bg_value"]
            scale = self.cfg["scale"]
            rg = (rg - bg) / scale + 0.5
            rg = np.clip(rg, 0.0, 1.0)
            t = torch.from_numpy(rg).permute(2, 0, 1)
        elif dt == "rgba_a":
            arr = np.array(img, dtype=np.float32)
            a = arr[:, :, 3:4] / self.cfg["max_val"]
            t = torch.from_numpy(a).permute(2, 0, 1)
        else:
            raise ValueError(f"Unknown dtype: {dt}")

        if t.shape[1] != self.img_size or t.shape[2] != self.img_size:
            t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        return t

    @staticmethod
    def _rotate_vel(target, k):
        """Apply velocity direction correction after spatial rot90.
        In normalized space center=0.5, negate means 1.0 - val.
        k=1 (90° CCW): (Vx,Vy) → (-Vy, Vx)
        k=2 (180°):    (Vx,Vy) → (-Vx,-Vy)
        k=3 (270° CCW):(Vx,Vy) → ( Vy,-Vx)
        """
        vx, vy = target[0].clone(), target[1].clone()
        if k == 1:
            target[0] = 1.0 - vy
            target[1] = vx
        elif k == 2:
            target[0] = 1.0 - vx
            target[1] = 1.0 - vy
        elif k == 3:
            target[0] = vy
            target[1] = 1.0 - vx
        return target

    def _random_crop(self, label, target, scale_range=(0.8, 1.0)):
        s = torch.empty(1).uniform_(*scale_range).item()
        crop_size = int(self.img_size * s)
        if crop_size < self.img_size:
            max_off = self.img_size - crop_size
            y = torch.randint(0, max_off + 1, (1,)).item()
            x = torch.randint(0, max_off + 1, (1,)).item()
            label = label[:, y:y + crop_size, x:x + crop_size]
            target = target[:, y:y + crop_size, x:x + crop_size]
            label = F.interpolate(label.unsqueeze(0), size=(self.img_size, self.img_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
            target = F.interpolate(target.unsqueeze(0), size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
        return label, target

    def __getitem__(self, idx):
        label = self._load_label(self.label_paths[idx])
        target = self._load_target(self.img_paths[idx])

        is_vel = self.target_type in ("vel", "vel_x25")

        if self.augment:
            if torch.rand(1).item() > 0.5:
                label = torch.flip(label, [-1])
                target = torch.flip(target, [-1])
                if is_vel:
                    target[0] = 1.0 - target[0]
            if torch.rand(1).item() > 0.5:
                label = torch.flip(label, [-2])
                target = torch.flip(target, [-2])
                if is_vel:
                    target[1] = 1.0 - target[1]
            k = torch.randint(0, 4, (1,)).item()
            if k:
                label = torch.rot90(label, k, [-2, -1])
                target = torch.rot90(target, k, [-2, -1])
                if is_vel:
                    target = self._rotate_vel(target, k)
            if torch.rand(1).item() > 0.5:
                label, target = self._random_crop(label, target)
            if torch.rand(1).item() > 0.3:
                jitter = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
                label = (label * jitter).clamp(0.0, 1.0)

        return label, target
