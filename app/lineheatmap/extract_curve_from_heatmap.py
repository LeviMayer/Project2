import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
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


def moving_average_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(np.float32)

    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def fill_nans_by_interp(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    xs = np.arange(len(out), dtype=np.float32)
    out[~valid] = np.interp(xs[~valid], xs[valid], out[valid]).astype(np.float32)
    return out


def soft_argmax_1d(values: np.ndarray, power: float = 2.0) -> Tuple[float, float]:
    """
    Returns:
      y_soft: soft argmax position
      conf: max value in column
    """
    conf = float(values.max())
    if conf <= 0:
        return np.nan, conf

    weights = np.maximum(values, 0.0) ** power
    denom = float(weights.sum())
    if denom <= 1e-8:
        return float(np.argmax(values)), conf

    y_coords = np.arange(len(values), dtype=np.float32)
    y_soft = float((weights * y_coords).sum() / denom)
    return y_soft, conf


def extract_curve_from_heatmap(
    heatmap: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    threshold: float = 0.05,
    smooth_window: int = 7,
    crop_to_plot_area: bool = True,
    plot_left_frac: float = 0.12,
    plot_bottom_frac: float = 0.14,
    plot_width_frac: float = 0.82,
    plot_height_frac: float = 0.78,
    peak_power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    heatmap: [H, W], values in [0,1]

    Returns:
      xs_chart: [N]
      ys_chart: [N]
      conf:     [N]
    """
    h, w = heatmap.shape

    if crop_to_plot_area:
        left = int(round(plot_left_frac * w))
        right = int(round((plot_left_frac + plot_width_frac) * w))
        top = int(round((1.0 - (plot_bottom_frac + plot_height_frac)) * h))
        bottom = int(round((1.0 - plot_bottom_frac) * h))

        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))

        cropped = heatmap[top:bottom, left:right]
    else:
        left, top = 0, 0
        cropped = heatmap

    ch, cw = cropped.shape

    ys_px_local = []
    conf = []

    for x in range(cw):
        col = cropped[:, x]
        y_soft, c = soft_argmax_1d(col, power=peak_power)

        if c < threshold:
            ys_px_local.append(np.nan)
        else:
            ys_px_local.append(y_soft)
        conf.append(c)

    ys_px_local = np.asarray(ys_px_local, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)

    ys_px_local = fill_nans_by_interp(ys_px_local)

    if np.isfinite(ys_px_local).any():
        ys_px_local = moving_average_1d(ys_px_local, smooth_window)
    else:
        ys_px_local[:] = ch / 2.0

    xs_px = np.arange(cw, dtype=np.float32) + float(left)
    ys_px = ys_px_local + float(top)

    # Map pixels back to chart coordinates using plot area only
    if crop_to_plot_area:
        plot_left_px = float(left)
        plot_right_px = float(right - 1)
        plot_top_px = float(top)
        plot_bottom_px = float(bottom - 1)
    else:
        plot_left_px = 0.0
        plot_right_px = float(w - 1)
        plot_top_px = 0.0
        plot_bottom_px = float(h - 1)

    xs_norm = (xs_px - plot_left_px) / max(plot_right_px - plot_left_px, 1e-8)
    xs_chart = x_min + xs_norm * (x_max - x_min)

    ys_norm = 1.0 - ((ys_px - plot_top_px) / max(plot_bottom_px - plot_top_px, 1e-8))
    ys_chart = y_min + ys_norm * (y_max - y_min)

    return xs_chart.astype(np.float32), ys_chart.astype(np.float32), conf.astype(np.float32)


def save_curve_csv(path: Path, xs: np.ndarray, ys: np.ndarray, conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "confidence"])
        for x, y, c in zip(xs, ys, conf):
            writer.writerow([float(x), float(y), float(c)])


def chart_to_pixel_coords(
    xs_chart: np.ndarray,
    ys_chart: np.ndarray,
    image_h: int,
    image_w: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_left_frac: float = 0.12,
    plot_bottom_frac: float = 0.14,
    plot_width_frac: float = 0.82,
    plot_height_frac: float = 0.78,
) -> Tuple[np.ndarray, np.ndarray]:
    plot_left = plot_left_frac * image_w
    plot_right = (plot_left_frac + plot_width_frac) * image_w

    plot_bottom = (1.0 - plot_bottom_frac) * image_h
    plot_top = (1.0 - (plot_bottom_frac + plot_height_frac)) * image_h

    xs_norm = (xs_chart - x_min) / max(x_max - x_min, 1e-8)
    ys_norm = (ys_chart - y_min) / max(y_max - y_min, 1e-8)

    xs_px = plot_left + xs_norm * (plot_right - plot_left)
    ys_px = plot_bottom - ys_norm * (plot_bottom - plot_top)

    return xs_px.astype(np.float32), ys_px.astype(np.float32)


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

    h, w = pred_heatmap.shape
    curve_x_px, curve_y_px = chart_to_pixel_coords(
        xs_chart,
        ys_chart,
        image_h=h,
        image_w=w,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

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
    axes[3].plot(curve_x_px, curve_y_px, linewidth=1.5)
    axes[3].set_title("overlay+curve")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lineheatmap/lineheatmap_vitl16.yaml", type=str)
    parser.add_argument("--checkpoint", default="outputs/lineheatmap_vitl16/best.pth", type=str)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--smooth_window", type=int, default=7)
    parser.add_argument("--peak_power", type=float, default=2.0)
    parser.add_argument("--no_crop_to_plot_area", action="store_true")
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
            crop_to_plot_area=not args.no_crop_to_plot_area,
            peak_power=args.peak_power,
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