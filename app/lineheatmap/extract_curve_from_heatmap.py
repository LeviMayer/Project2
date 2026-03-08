import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

from app.lineheatmap.dataset import LineHeatmapDataset
from app.lineheatmap.model import LineHeatmapModel
from app.vjepa.utils import init_video_model


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: Dict[str, Any], device: torch.device, checkpoint_path: str) -> torch.nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    encoder, _predictor = init_video_model(
        uniform_power=model_cfg.get("uniform_power", True),
        use_mask_tokens=model_cfg.get("use_mask_tokens", True),
        num_mask_tokens=1,
        zero_init_mask_tokens=model_cfg.get("zero_init_mask_tokens", True),
        device=device,
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg.get("num_frames", 1),
        tubelet_size=data_cfg.get("tubelet_size", 1),
        model_name=model_cfg["model_name"],
        crop_size=data_cfg["crop_size"],
        pred_depth=model_cfg.get("pred_depth", 12),
        pred_embed_dim=model_cfg.get("pred_embed_dim", 384),
        use_sdpa=cfg.get("meta", {}).get("use_sdpa", True),
    )

    embed_dim_map = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_huge": 1280,
    }
    embed_dim = train_cfg.get("embed_dim", embed_dim_map.get(model_cfg["model_name"], 1024))

    model = LineHeatmapModel(
        encoder=encoder,
        embed_dim=embed_dim,
        image_size=data_cfg["crop_size"],
        patch_size=data_cfg["patch_size"],
        out_channels=1,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def extract_curve_from_heatmap(
    heatmap: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    threshold: float = 0.05,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    heatmap: [H, W], values in [0,1]
    returns:
      xs_chart: [W]
      ys_chart: [W]
      conf:     [W]
    """
    h, w = heatmap.shape
    ys_px = []
    conf = []

    for x in range(w):
        col = heatmap[:, x]
        y = int(np.argmax(col))
        c = float(col[y])

        if c < threshold:
            ys_px.append(np.nan)
        else:
            ys_px.append(float(y))
        conf.append(c)

    ys_px = np.asarray(ys_px, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    # fill NaNs by interpolation
    valid = np.isfinite(ys_px)
    if valid.any():
        xs_idx = np.arange(w, dtype=np.float32)
        ys_px = np.interp(xs_idx, xs_idx[valid], ys_px[valid]).astype(np.float32)
    else:
        ys_px[:] = h // 2

    # optional smoothing
    if smooth_window > 1:
        pad = smooth_window // 2
        padded = np.pad(ys_px, (pad, pad), mode="edge")
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        ys_px = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    xs_px = np.arange(w, dtype=np.float32)

    xs_chart = x_min + (xs_px / max(w - 1, 1)) * (x_max - x_min)
    ys_norm = 1.0 - (ys_px / max(h - 1, 1))
    ys_chart = y_min + ys_norm * (y_max - y_min)

    return xs_chart, ys_chart, conf


def save_curve_csv(path: Path, xs: np.ndarray, ys: np.ndarray, conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "confidence"])
        for x, y, c in zip(xs, ys, conf):
            writer.writerow([float(x), float(y), float(c)])


def save_debug_figure(
    out_path: Path,
    image_t: torch.Tensor,
    gt_heatmap_t: torch.Tensor,
    pred_heatmap: np.ndarray,
    xs_chart: np.ndarray,
    ys_chart: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = image_t.cpu().permute(1, 2, 0).numpy()
    gt_heatmap = gt_heatmap_t.cpu().squeeze(0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(gt_heatmap, cmap="hot")
    axes[1].set_title("gt_heatmap")
    axes[1].axis("off")

    axes[2].imshow(pred_heatmap, cmap="hot")
    axes[2].set_title("pred_heatmap")
    axes[2].axis("off")

    axes[3].imshow(image)
    axes[3].imshow(pred_heatmap, cmap="hot", alpha=0.45)
    axes[3].plot(
        (xs_chart - x_min) / max(x_max - x_min, 1e-8) * (pred_heatmap.shape[1] - 1),
        (1.0 - (ys_chart - y_min) / max(y_max - y_min, 1e-8)) * (pred_heatmap.shape[0] - 1),
        linewidth=1.0,
    )
    axes[3].set_title("overlay+curve")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lineheatmap/lineheatmap_vitl16.yaml", type=str)
    parser.add_argument("--checkpoint",default="outputs/lineheatmap_vitl16/best.pth", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="outputs/extracted_curves")
    parser.add_argument("--save_debug", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    root_dir = data_cfg["root_dir"]
    manifest = data_cfg["val_manifest"] if args.split == "val" else data_cfg["train_manifest"]

    dataset = LineHeatmapDataset(
        root_dir=root_dir,
        manifest_path=manifest,
        image_key=data_cfg.get("image_key", "full"),
        image_size=data_cfg["crop_size"],
        heatmap_size=data_cfg.get("heatmap_size", data_cfg["crop_size"]),
        sigma=data_cfg.get("sigma", 2.5),
        x_min=data_cfg.get("x_min", 0.0),
        x_max=data_cfg.get("x_max", 10.0),
        y_min=data_cfg.get("y_min", 0.0),
        y_max=data_cfg.get("y_max", 100.0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device, args.checkpoint)

    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    dbg_dir = out_dir / "debug"

    end_index = min(args.start_index + args.num_samples, len(dataset))
    print(f"[INFO] split={args.split} dataset_size={len(dataset)}")
    print(f"[INFO] processing samples [{args.start_index}, {end_index})")

    x_min = data_cfg.get("x_min", 0.0)
    x_max = data_cfg.get("x_max", 10.0)
    y_min = data_cfg.get("y_min", 0.0)
    y_max = data_cfg.get("y_max", 100.0)

    for idx in range(args.start_index, end_index):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_heatmap = sample["heatmap"]
        sample_id = sample["id"]

        logits = model(image)
        pred_heatmap = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        xs_chart, ys_chart, conf = extract_curve_from_heatmap(
            pred_heatmap,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
        )

        csv_path = csv_dir / f"{sample_id}.csv"
        save_curve_csv(csv_path, xs_chart, ys_chart, conf)

        if args.save_debug:
            dbg_path = dbg_dir / f"{sample_id}.png"
            save_debug_figure(
                dbg_path,
                sample["image"],
                gt_heatmap,
                pred_heatmap,
                xs_chart,
                ys_chart,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

        print(f"[INFO] wrote {csv_path}")

    print("[INFO] done")


if __name__ == "__main__":
    main()