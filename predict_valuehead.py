#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.vjepa.utils import init_video_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ValueHead(nn.Module):
    """
    Matches your trained checkpoint (fc.weight, fc.bias).
    Predicts ONLY y-values for K fixed x-positions in [0,1].
    """
    def __init__(self, embed_dim: int, K: int, pool: str = "cls"):
        super().__init__()
        self.K = int(K)
        self.pool = pool  # "cls" or "mean"
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, K),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            feat = tokens.mean(dim=1)   # [B, D]
        else:
            feat = tokens[:, 0]         # [B, D]
        y = torch.sigmoid(self.fc(feat))  # [B, K] in [0,1]
        return y


def load_image_224_rgb(path: Path, crop_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((crop_size, crop_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,C]
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    return torch.from_numpy(arr)


def maybe_overlay(out_png: Path, img_path: Path, xs: np.ndarray, ys: np.ndarray, crop_size: int = 224):
    """
    Draw predicted curve on image. xs, ys in [0,1].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(img_path).convert("RGB").resize((crop_size, crop_size), Image.BICUBIC)

    px_x = xs * (crop_size - 1)
    px_y = (1.0 - ys) * (crop_size - 1)  # y=0 bottom in plot coords

    plt.figure(figsize=(crop_size/100, crop_size/100), dpi=100)
    plt.imshow(img)
    plt.plot(px_x, px_y, linewidth=2)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_jepa_encoder(encoder: nn.Module, jepa_ckpt_path: Path):
    ckpt = torch.load(jepa_ckpt_path, map_location="cpu")
    if "encoder" not in ckpt:
        raise KeyError(f"JEPA checkpoint keys={list(ckpt.keys())}, expected key 'encoder'")
    enc_sd = strip_module_prefix(ckpt["encoder"])
    msg = encoder.load_state_dict(enc_sd, strict=True)
    print("Loaded encoder:", msg)


def load_valuehead(head_ckpt_path: Path, embed_dim: int, device: torch.device) -> Tuple[ValueHead, int, str]:
    head_ckpt = torch.load(head_ckpt_path, map_location="cpu")

    # Your trainer saved a dict with "valuehead" and metadata
    if isinstance(head_ckpt, dict) and "valuehead" in head_ckpt:
        head_sd = head_ckpt["valuehead"]
        pool = str(head_ckpt.get("pool", "cls"))
    else:
        head_sd = head_ckpt
        pool = "cls"

    head_sd = strip_module_prefix(head_sd)

    if "fc.weight" not in head_sd:
        raise KeyError(f"Head checkpoint missing 'fc.weight'. Keys={list(head_sd.keys())[:20]} ...")

    K = int(head_sd["fc.weight"].shape[0])

    head = ValueHead(embed_dim=embed_dim, K=K, pool=pool).to(device)
    msg = head.load_state_dict(head_sd, strict=True)
    print("Loaded head:", msg, f"(K={K}, pool={pool})")
    head.eval()
    return head, K, pool


def read_manifest(manifest_path: Path, data_dir: Optional[Path] = None) -> List[Dict]:
    items = []
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            # manifest usually stores relative paths like "images/xxx.png"
            if data_dir is not None:
                if j.get("full"):
                    j["full_abs"] = str((data_dir / j["full"]).resolve())
                if j.get("masked"):
                    j["masked_abs"] = str((data_dir / j["masked"]).resolve())
                if j.get("csv"):
                    j["csv_abs"] = str((data_dir / j["csv"]).resolve())
            items.append(j)
    return items


@torch.no_grad()
def predict_one(
    encoder: nn.Module,
    head: nn.Module,
    img_path: Path,
    device: torch.device,
    crop_size: int
) -> np.ndarray:
    x = load_image_224_rgb(img_path, crop_size=crop_size).unsqueeze(0).to(device)  # [1,3,H,W]
    tokens = encoder(x)            # [1, N, D]
    y = head(tokens)[0]            # [K]
    return y.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", type=str, required=True)
    ap.add_argument("--head_ckpt", type=str, required=True)

    # Option A: single image
    ap.add_argument("--image", type=str, default=None, help="single image path (optional)")

    # Option B: batch from manifest
    ap.add_argument("--manifest", type=str, default=None, help="manifest.jsonl (optional)")
    ap.add_argument("--data_dir", type=str, default=None, help="root dir that contains manifest relative paths")

    ap.add_argument("--out_dir", type=str, default="preds_out")
    ap.add_argument("--n", type=int, default=100, help="how many images to sample from manifest (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="predict all images from manifest")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--write_per_image_csv", action="store_true", help="also write one csv per image")
    ap.add_argument("--overlay", action="store_true", help="write overlay png per image (slow)")

    ap.add_argument("--model_name", type=str, default="vit_large")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--crop_size", type=int, default=224)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--tubelet_size", type=int, default=1)
    ap.add_argument("--use_sdpa", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build encoder
    encoder, _ = init_video_model(
        device=device,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        model_name=args.model_name,
        crop_size=args.crop_size,
        pred_depth=1,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=args.use_sdpa,
    )
    encoder.eval()

    # Load JEPA
    load_jepa_encoder(encoder, Path(args.jepa_ckpt))

    # Load head (infer K)
    embed_dim = encoder.backbone.embed_dim
    head, K, pool = load_valuehead(Path(args.head_ckpt), embed_dim=embed_dim, device=device)

    xs = np.linspace(0.0, 1.0, K)

    # -------- single image mode --------
    if args.image is not None:
        img_path = Path(args.image)
        y = predict_one(encoder, head, img_path, device, crop_size=args.crop_size)
        out_csv = out_dir / "pred_points.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_norm", "y_norm"])
            for x0, y0 in zip(xs, y):
                w.writerow([float(x0), float(y0)])
        print("Wrote:", out_csv)

        if args.overlay:
            out_png = out_dir / "overlay.png"
            maybe_overlay(out_png, img_path, xs, y, crop_size=args.crop_size)
            print("Wrote:", out_png)
        return

    # -------- manifest batch mode --------
    if args.manifest is None:
        raise SystemExit("Provide either --image OR --manifest (and usually --data_dir).")

    manifest_path = Path(args.manifest)
    data_dir = Path(args.data_dir) if args.data_dir is not None else None
    items = read_manifest(manifest_path, data_dir=data_dir)

    # choose image paths
    img_paths = []
    for it in items:
        p = it.get("full_abs") or it.get("full")
        if p is None:
            continue
        img_paths.append(Path(p))

    if len(img_paths) == 0:
        raise RuntimeError("No images found from manifest. Check --data_dir and manifest paths.")

    if args.all:
        chosen = img_paths
    else:
        n = min(args.n, len(img_paths))
        chosen = random.sample(img_paths, n)

    # main output CSV (one row per predicted point)
    out_big = out_dir / "predictions.csv"
    per_img_dir = out_dir / "per_image_csv"
    overlay_dir = out_dir / "overlays"
    if args.write_per_image_csv:
        per_img_dir.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    with out_big.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "k", "x_norm", "y_norm"])

        for i, img_path in enumerate(chosen, 1):
            y = predict_one(encoder, head, img_path, device, crop_size=args.crop_size)

            # write to big file
            for k_idx, (x0, y0) in enumerate(zip(xs, y)):
                w.writerow([str(img_path), int(k_idx), float(x0), float(y0)])

            # optional per-image csv
            if args.write_per_image_csv:
                out_csv = per_img_dir / (img_path.stem + "_pred.csv")
                with out_csv.open("w", newline="") as ff:
                    ww = csv.writer(ff)
                    ww.writerow(["x_norm", "y_norm"])
                    for x0, y0 in zip(xs, y):
                        ww.writerow([float(x0), float(y0)])

            # optional overlay
            if args.overlay:
                out_png = overlay_dir / (img_path.stem + "_overlay.png")
                maybe_overlay(out_png, img_path, xs, y, crop_size=args.crop_size)

            if i % 50 == 0 or i == len(chosen):
                print(f"[{i}/{len(chosen)}] done")

    print("Wrote:", out_big)
    print("Done.")


if __name__ == "__main__":
    main()