#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --- your repo imports (assumes you run from repo root) ---
from app.vjepa.utils import init_video_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ValueHead(nn.Module):
    """
    Simple head: takes CLS token embedding and predicts K (x,y) pairs.
    Output is normalized to [0,1] via sigmoid.
    """
    def __init__(self, embed_dim: int, K: int = 64, hidden: int = 1024):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * K),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D] coming from encoder (includes CLS at tokens[:,0])
        returns: [B, K, 2] in [0,1]
        """
        cls = tokens[:, 0]  # [B, D]
        out = self.mlp(cls)  # [B, 2K]
        out = out.view(-1, self.K, 2)
        out = torch.sigmoid(out)
        return out


def load_image_224_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,C]
    # normalize
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # [C,H,W]
    return torch.from_numpy(arr)  # float32


def save_pred_csv(out_csv: Path, pts_xy: np.ndarray):
    # pts_xy: [K,2] in [0,1]
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_norm", "y_norm"])
        for x, y in pts_xy:
            w.writerow([float(x), float(y)])


def save_overlay(out_png: Path, img_path: Path, pts_xy: np.ndarray):
    """
    Draw predicted curve on the image in pixel space.
    pts_xy in [0,1] with x increasing ideally.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    xs = pts_xy[:, 0] * 223.0
    ys = (1.0 - pts_xy[:, 1]) * 223.0  # y=0 bottom

    plt.figure(figsize=(224/100, 224/100), dpi=100)
    plt.imshow(img)
    plt.plot(xs, ys, linewidth=2)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_ckpt", type=str, required=True, help="JEPA checkpoint .pth.tar (contains key 'encoder')")
    ap.add_argument("--head_ckpt", type=str, required=True, help="ValueHead checkpoint .pth (state_dict)")
    ap.add_argument("--image", type=str, required=True, help="input image (png/jpg)")
    ap.add_argument("--out_dir", type=str, default="pred_out", help="output folder")
    ap.add_argument("--K", type=int, default=64, help="number of points to predict")
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

    # 1) Build encoder (predictor unused here)
    encoder, _predictor = init_video_model(
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

    # 2) Load JEPA encoder weights
    ckpt = torch.load(args.jepa_ckpt, map_location="cpu")
    if "encoder" not in ckpt:
        raise KeyError(f"Checkpoint has keys {list(ckpt.keys())}, expected key 'encoder'")
    msg = encoder.load_state_dict(ckpt["encoder"], strict=True)
    print("Loaded encoder:", msg)

    # 3) Build + load head
    embed_dim = encoder.backbone.embed_dim  # VisionTransformer embed dim
    head = ValueHead(embed_dim=embed_dim, K=args.K).to(device)
    head_sd = torch.load(args.head_ckpt, map_location="cpu")
    head.load_state_dict(head_sd, strict=True)
    head.eval()

    # 4) Load image
    x = load_image_224_rgb(Path(args.image)).unsqueeze(0).to(device)  # [1,3,224,224]

    # 5) Forward
    with torch.no_grad():
        tokens = encoder(x)          # [B, N, D]
        pred = head(tokens)[0]       # [K,2]
        pred_np = pred.detach().cpu().numpy()

    # sort by x (helps make a clean curve)
    pred_np = pred_np[np.argsort(pred_np[:, 0])]

    # 6) Save outputs
    out_csv = out_dir / "pred_points.csv"
    out_png = out_dir / "overlay.png"
    save_pred_csv(out_csv, pred_np)
    save_overlay(out_png, Path(args.image), pred_np)

    print("Wrote:", out_csv)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()