import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import LineHeatmapDataset


def save_debug_figure(image, heatmap, sample_id: str, out_path: Path) -> None:
    """
    image: torch tensor [3,H,W] in [0,1]
    heatmap: torch tensor [1,H,W] in [0,1]
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()
    hm_np = heatmap.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(hm_np, cmap="hot")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(hm_np, cmap="hot", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(sample_id)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="full")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--heatmap_size", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=10.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=100.0)
    parser.add_argument("--out", type=str, default="outputs/lineheatmap_debug/debug_sample.png")
    args = parser.parse_args()

    ds = LineHeatmapDataset(
        root_dir=args.root_dir,
        manifest_path=args.manifest,
        image_key=args.image_key,
        image_size=args.image_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    if len(ds) == 0:
        raise RuntimeError("Dataset is empty.")

    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    image = sample["image"]
    heatmap = sample["heatmap"]
    sample_id = sample["id"]

    print(f"[INFO] dataset size: {len(ds)}")
    print(f"[INFO] sample index: {idx}")
    print(f"[INFO] sample id: {sample_id}")
    print(f"[INFO] image shape: {tuple(image.shape)}")
    print(f"[INFO] heatmap shape: {tuple(heatmap.shape)}")
    print(f"[INFO] heatmap min/max: {float(heatmap.min()):.4f} / {float(heatmap.max()):.4f}")

    out_path = Path(args.out)
    save_debug_figure(image, heatmap, sample_id, out_path)
    print(f"[INFO] wrote debug figure to: {out_path}")


if __name__ == "__main__":
    main()
