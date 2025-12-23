# save as: make_aug_dir.py
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unnormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    # x: (3,H,W) normalized
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x * std_t + mean_t).clamp(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subset-root",
        type=Path,
        required=True,
        help="Path like .../datasets/VRGVMH_cifar10_subset (contains train/ and test/)",
    )
    ap.add_argument(
        "--dst-name",
        type=str,
        default="train_aug",
        help="Folder created under subset-root to store augmented images",
    )
    ap.add_argument("--n-aug", type=int, default=8, help="Augmented views per image")
    ap.add_argument("--imgsz", type=int, default=32)
    ap.add_argument(
        "--auto-augment",
        type=str,
        default="randaugment",
        choices=["none", "randaugment", "augmix", "autoaugment"],
    )
    ap.add_argument("--erasing", type=float, default=0.0)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--vflip", type=float, default=0.0)
    ap.add_argument("--hsv-h", type=float, default=0.015)
    ap.add_argument("--hsv-s", type=float, default=0.4)
    ap.add_argument("--hsv-v", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-orig", action="store_true", help="Also copy originals into dst")
    args = ap.parse_args()

    set_seed(args.seed)

    subset_root = args.subset_root
    src_train = subset_root / "train"
    dst_train = subset_root / args.dst_name  # will contain class subfolders
    dst_train.mkdir(parents=True, exist_ok=True)

    # Ultralytics classification augmentation pipeline (same style as training)
    from ultralytics.data.augment import classify_augmentations

    # Mean/std are used inside the pipeline; we unnormalize before saving PNGs
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aa = None if args.auto_augment == "none" else args.auto_augment
    aug_tfm = classify_augmentations(
        size=args.imgsz,
        mean=mean,
        std=std,
        auto_augment=aa,
        erasing=args.erasing,
        hflip=args.hflip,
        vflip=args.vflip,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for cls_dir in sorted([p for p in src_train.iterdir() if p.is_dir()]):
        dst_cls = dst_train / cls_dir.name
        dst_cls.mkdir(parents=True, exist_ok=True)

        for img_path in sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in exts]):
            if args.copy_orig:
                (dst_cls / img_path.name).write_bytes(img_path.read_bytes())

            img = Image.open(img_path).convert("RGB")
            stem = img_path.stem

            for j in range(args.n_aug):
                x = aug_tfm(img)                      # normalized tensor (3,H,W)
                x = unnormalize(x, mean, std)         # back to [0,1] for saving
                out_path = dst_cls / f"{stem}__aug{j:02d}.png"
                torchvision.utils.save_image(x, out_path)

    print(f"Augmented images written to: {dst_train}")
    print("If you later train Ultralytics on that folder, delete any *.cache so it re-indexes.")


if __name__ == "__main__":
    main()
