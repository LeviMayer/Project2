#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Minimal image preprocessing (like ImageNet-style)
# ----------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_image_rgb_224(path: Path, size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H W C
    x = torch.from_numpy(arr).permute(2, 0, 1)        # C H W
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


# ----------------------------
# A simple default ValueHead
# (If your training used a different head, adjust here.)
# Input: token features [B, N, D] or pooled [B, D]
# Output: [B, Lmax, T]
# ----------------------------
class ValueHead(nn.Module):
    def __init__(self, in_dim: int, T: int, Lmax: int = 1, hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.T = T
        self.Lmax = Lmax
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, Lmax * T),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, D] or [B, N, D]
        if feats.ndim == 3:
            feats = feats.mean(dim=1)  # pool tokens -> [B, D]
        y = self.mlp(feats)            # [B, L*T]
        y = y.view(feats.size(0), self.Lmax, self.T)
        return y


# ----------------------------
# Load encoder from your repo (Meta JEPA code)
# ----------------------------
def build_encoder_from_repo(model_name: str, crop_size: int, patch_size: int, num_frames: int, tubelet_size: int, use_sdpa: bool, device: torch.device):
    # expects to run inside repo root so imports work
    from app.vjepa.utils import init_video_model
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=1,          # predictor not needed; keep tiny
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=use_sdpa,
    )
    # we only need encoder
    del predictor
    encoder.eval()
    return encoder


def _extract_state_dict(ckpt):
    # supports various checkpoint formats
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        return ckpt
    raise ValueError("Unsupported checkpoint type")


def load_encoder_and_head(
    ckpt_path: Path,
    device: torch.device,
    model_name: str,
    crop_size: int,
    patch_size: int,
    num_frames: int,
    tubelet_size: int,
    use_sdpa: bool,
    T: int,
    Lmax: int,
    head_hidden: int,
):
    """
    Tries to load:
      - encoder weights from ckpt if present, else uses freshly built encoder
      - head weights from ckpt if present, else keeps randomly init head
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    encoder = build_encoder_from_repo(
        model_name=model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=use_sdpa,
        device=device,
    )

    # Find encoder embed dim
    # encoder is MultiMaskWrapper(backbone); embed_dim on backbone
    embed_dim = encoder.backbone.embed_dim

    head = ValueHead(in_dim=embed_dim, T=T, Lmax=Lmax, hidden=head_hidden).to(device).eval()

    # Try to load encoder.* keys
    enc_candidates = {}
    head_candidates = {}

    # common patterns:
    #  - encoder.backbone.* or encoder.* or backbone.*
    #  - head.* or value_head.* or reg_head.* etc.
    for k, v in sd.items():
        ks = k
        if ks.startswith("module."):
            ks = ks[len("module."):]
        # heuristics
        if ks.startswith("encoder."):
            enc_candidates[ks[len("encoder."):]] = v
        elif ks.startswith("backbone."):
            # might be pure backbone weights
            enc_candidates["backbone." + ks[len("backbone."):]] = v
        elif ks.startswith("head."):
            head_candidates[ks[len("head."):]] = v
        elif ks.startswith("value_head."):
            head_candidates[ks[len("value_head."):]] = v
        elif ks.startswith("reg_head."):
            head_candidates[ks[len("reg_head."):]] = v

    # Another common case: keys already match encoder/backbone modules
    # Try direct load if no enc_candidates found
    loaded_any = False

    if len(enc_candidates) > 0:
        msg = encoder.load_state_dict(enc_candidates, strict=False)
        print("[load] encoder from encoder.* keys:", msg)
        loaded_any = True
    else:
        # try best-effort direct load
        msg = encoder.load_state_dict(sd, strict=False)
        print("[load] encoder direct (best-effort):", msg)

    if len(head_candidates) > 0:
        msg = head.load_state_dict(head_candidates, strict=False)
        print("[load] head from head/value_head keys:", msg)
        loaded_any = True
    else:
        # try to load from sd directly (maybe head keys match)
        msg = head.load_state_dict(sd, strict=False)
        print("[load] head direct (best-effort):", msg)

    encoder.to(device).eval()
    head.to(device).eval()

    return encoder, head


@torch.no_grad()
def predict_one(
    encoder,
    head,
    img_path: Path,
    device: torch.device,
    img_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    assume_normalized_y: bool,
    out_csv: Path,
    out_overlay: Path | None,
):
    x = load_image_rgb_224(img_path, size=img_size).unsqueeze(0).to(device)  # [1,3,H,W]

    # encoder.backbone expects [B,3,H,W] for image-like
    feats = encoder.backbone(x)  # typically [B, N, D]
    y_pred = head(feats)[0].detach().cpu().numpy()  # [Lmax, T]

    # Scale y if model predicts 0..1
    if assume_normalized_y:
        y_pred = y_min + (y_max - y_min) * y_pred

    T = y_pred.shape[1]
    xs = np.linspace(x_min, x_max, T)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("x,y,line_id\n")
        for li in range(y_pred.shape[0]):
            for xi, yi in zip(xs, y_pred[li]):
                f.write(f"{float(xi)},{float(yi)},{li}\n")

    if out_overlay is not None:
        out_overlay.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
        fig = plt.figure(figsize=(img_size/100, img_size/100), dpi=100)
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(img)
        ax.axis("off")

        # simple overlay: draw predicted curve in image coordinates assuming axes fill most area
        # This is only a rough debug visualization.
        # If you want pixel-perfect axis mapping, we do a separate "axis detection" step later.
        pad = int(0.12 * img_size)
        W = img_size
        H = img_size
        x0, x1 = pad, W - pad
        y0, y1 = pad, H - pad

        def map_xy(xv, yv):
            # map data coords -> image coords
            u = x0 + (xv - x_min) / (x_max - x_min + 1e-9) * (x1 - x0)
            v = y1 - (yv - y_min) / (y_max - y_min + 1e-9) * (y1 - y0)
            return u, v

        for li in range(y_pred.shape[0]):
            pts = [map_xy(xv, yv) for xv, yv in zip(xs, y_pred[li])]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2)

        fig.savefig(out_overlay, dpi=100)
        plt.close(fig)


def iter_images(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for p in sorted(input_path.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to valuehead checkpoint (.pth/.pt)")
    ap.add_argument("--input", type=str, required=True, help="Image file or directory")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (CSVs + overlays)")

    # model params (must match training)
    ap.add_argument("--model_name", type=str, default="vit_large")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--tubelet_size", type=int, default=1)
    ap.add_argument("--use_sdpa", action="store_true")

    # head params
    ap.add_argument("--T", type=int, default=60, help="Number of points per line")
    ap.add_argument("--Lmax", type=int, default=1, help="Max number of lines predicted")
    ap.add_argument("--head_hidden", type=int, default=1024)

    # axis ranges (data coords)
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    ap.add_argument("--assume_normalized_y", action="store_true",
                    help="If set: interpret model outputs as 0..1 and scale to [y_min,y_max].")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overlay", action="store_true", help="Save debug overlay PNG per image")

    args = ap.parse_args()

    device = torch.device(args.device)

    # IMPORTANT: run from repo root so imports work
    # (or set PYTHONPATH accordingly)
    encoder, head = load_encoder_and_head(
        ckpt_path=Path(args.ckpt),
        device=device,
        model_name=args.model_name,
        crop_size=args.img_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        use_sdpa=args.use_sdpa,
        T=args.T,
        Lmax=args.Lmax,
        head_hidden=args.head_hidden,
    )

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_csv_dir = out_dir / "csv"
    out_ov_dir  = out_dir / "overlay"

    n = 0
    for img_path in iter_images(in_path):
        rel = img_path.name
        stem = img_path.stem
        out_csv = out_csv_dir / f"{stem}.csv"
        out_ov = (out_ov_dir / f"{stem}.png") if args.overlay else None

        predict_one(
            encoder=encoder,
            head=head,
            img_path=img_path,
            device=device,
            img_size=args.img_size,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            assume_normalized_y=args.assume_normalized_y,
            out_csv=out_csv,
            out_overlay=out_ov,
        )
        n += 1
        if n % 50 == 0:
            print(f"[{n}] done")

    print(f"Done. Wrote {n} CSVs to: {out_csv_dir}")
    if args.overlay:
        print(f"Overlays in: {out_ov_dir}")


if __name__ == "__main__":
    main()
