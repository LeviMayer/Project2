import os
import json
import math
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------
# Manifest / CSV loading
# --------------------------------------------------

def read_manifest(manifest_path: str) -> List[Dict]:
    data_root = os.path.dirname(os.path.abspath(manifest_path))

    def _fix(p):
        if p is None:
            return None
        p = p.replace("\\", "/")
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        return os.path.normpath(p)

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            it["full"] = _fix(it.get("full"))
            it["masked"] = _fix(it.get("masked"))
            it["csv"] = _fix(it.get("csv"))
            items.append(it)
    return items


def load_gt_csv(csv_path: str, line_id: Optional[int] = 0) -> List[Tuple[float, float]]:
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue

            try:
                lid = int(row.get("line_id", 0))
            except (TypeError, ValueError):
                lid = 0

            if line_id is not None and lid != int(line_id):
                continue

            points.append((x, y))

    points = sorted(points, key=lambda t: t[0])
    return points


def load_reconstructed_csv(csv_path: str) -> List[Tuple[float, float]]:
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x, y))

    points = sorted(points, key=lambda t: t[0])
    return points


def load_reconstructed_csv_with_pixels(csv_path: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Returns:
        data_points: [(x, y), ...]
        pixel_points: [(x_px, y_px), ...] if available, else []
    """
    data_points = []
    pixel_points = []
    has_pixel_cols = True

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                data_points.append((x, y))
            except (KeyError, TypeError, ValueError):
                continue

            try:
                x_px = float(row["x_px"])
                y_px = float(row["y_px"])
                pixel_points.append((x_px, y_px))
            except (KeyError, TypeError, ValueError):
                has_pixel_cols = False

    data_points = sorted(data_points, key=lambda t: t[0])

    if not has_pixel_cols or len(pixel_points) != len(data_points):
        pixel_points = []

    return data_points, pixel_points


# --------------------------------------------------
# Pixel GT loading from manifest
# --------------------------------------------------

def manifest_has_points_px(item: Dict) -> bool:
    pts = item.get("points_px", None)
    return isinstance(pts, list) and len(pts) > 0


def load_gt_points_px_from_manifest(item: Dict, line_id: Optional[int] = 0) -> List[Tuple[float, float]]:
    pts = item.get("points_px", [])
    out = []

    for p in pts:
        try:
            x = float(p["x"])
            y = float(p["y"])
            lid = int(p.get("line_id", 0))
        except (KeyError, TypeError, ValueError):
            continue

        if line_id is not None and lid != int(line_id):
            continue

        out.append((x, y))

    out = sorted(out, key=lambda t: t[0])
    return out


# --------------------------------------------------
# Matching
# --------------------------------------------------

def euclidean_data(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def greedy_match_data(
    pred_points: List[Tuple[float, float]],
    gt_points: List[Tuple[float, float]],
    x_thresh: float,
    y_thresh: float,
):
    candidates = []
    for pi, pp in enumerate(pred_points):
        for gi, gp in enumerate(gt_points):
            dx = abs(pp[0] - gp[0])
            dy = abs(pp[1] - gp[1])

            if dx <= x_thresh and dy <= y_thresh:
                d = euclidean_data(pp, gp)
                candidates.append((d, dx, dy, pi, gi))

    candidates.sort(key=lambda t: t[0])

    matched_pred = set()
    matched_gt = set()
    matches = []

    for d, dx, dy, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, d, dx, dy))

    tp = len(matches)
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    mean_abs_x_error = float(np.mean([m[3] for m in matches])) if matches else None
    mean_abs_y_error = float(np.mean([m[4] for m in matches])) if matches else None
    mean_euclidean_error = float(np.mean([m[2] for m in matches])) if matches else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_x_error": mean_abs_x_error,
        "mean_abs_y_error": mean_abs_y_error,
        "mean_euclidean_error": mean_euclidean_error,
        "matches": matches,
    }


# --------------------------------------------------
# Visualization helpers
# --------------------------------------------------

def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], r: int = 4):
    x = int(round(x))
    y = int(round(y))
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)


def draw_circle(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], r: int = 4):
    x = int(round(x))
    y = int(round(y))
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def render_overlay_image(
    image_path: str,
    gt_points_px: List[Tuple[float, float]],
    pred_points_px: List[Tuple[float, float]],
    out_path: str,
    connect_lines: bool = True,
):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # GT in green
    if connect_lines and len(gt_points_px) >= 2:
        draw.line([(float(x), float(y)) for x, y in gt_points_px], fill=(0, 255, 0), width=2)
    for x, y in gt_points_px:
        draw_cross(draw, x, y, color=(0, 255, 0), r=4)

    # Prediction in red
    if connect_lines and len(pred_points_px) >= 2:
        draw.line([(float(x), float(y)) for x, y in pred_points_px], fill=(255, 0, 0), width=2)
    for x, y in pred_points_px:
        draw_circle(draw, x, y, color=(255, 0, 0), r=4)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def save_example_visualizations(
    per_sample: List[Dict],
    examples_dir: str,
    num_best: int = 3,
    num_worst: int = 3,
    num_regular: int = 3,
):
    if len(per_sample) == 0:
        return

    os.makedirs(examples_dir, exist_ok=True)

    valid = [s for s in per_sample if s.get("image_path") is not None]

    if len(valid) == 0:
        return

    best_sorted = sorted(
        valid,
        key=lambda s: (
            -(s.get("f1", 0.0)),
            s.get("mean_euclidean_error") if s.get("mean_euclidean_error") is not None else 1e9,
        ),
    )[:num_best]

    worst_sorted = sorted(
        valid,
        key=lambda s: (
            s.get("f1", 0.0),
            -(s.get("fn", 0)),
            -(s.get("fp", 0)),
        ),
    )[:num_worst]

    step = max(1, len(valid) // max(1, num_regular))
    regular = [valid[i] for i in range(0, len(valid), step)[:num_regular]]

    groups = [
        ("best", best_sorted),
        ("worst", worst_sorted),
        ("examples", regular),
    ]

    for group_name, samples in groups:
        group_dir = os.path.join(examples_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        for rank, sample in enumerate(samples, start=1):
            sample_id = sample["id"]
            out_path = os.path.join(
                group_dir,
                f"{rank:02d}_{sample_id}_f1_{sample['f1']:.3f}.png"
            )
            render_overlay_image(
                image_path=sample["image_path"],
                gt_points_px=sample["gt_points_px"],
                pred_points_px=sample["pred_points_px"],
                out_path=out_path,
                connect_lines=True,
            )


# --------------------------------------------------
# Data reconstruction -> pixel conversion
# --------------------------------------------------

def pixel_to_data_with_bbox(
    px: float,
    py: float,
    plot_bbox_px: List[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float]:
    x0, y0, w, h = plot_bbox_px

    x_norm = (px - x0) / max(w, 1e-8)
    y_norm = (py - y0) / max(h, 1e-8)

    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    data_x = x_min + x_norm * (x_max - x_min)
    data_y = y_max - y_norm * (y_max - y_min)
    return float(data_x), float(data_y)


def data_y_to_pixel_with_bbox(
    y: float,
    plot_bbox_px: List[float],
    y_min: float,
    y_max: float,
) -> float:
    _, y0, _, h = plot_bbox_px
    y_norm = (y_max - y) / max(y_max - y_min, 1e-8)
    return float(y0 + y_norm * h)


def data_to_pixel_numeric_with_bbox(
    x: float,
    y: float,
    plot_bbox_px: List[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float]:
    x0, _, w, _ = plot_bbox_px
    x_norm = (x - x_min) / max(x_max - x_min, 1e-8)
    x_px = x0 + x_norm * w
    y_px = data_y_to_pixel_with_bbox(y, plot_bbox_px, y_min, y_max)
    return float(x_px), float(y_px)


def data_to_pixel_sample_aware(
    x: float,
    y: float,
    item: Dict,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float]:
    """
    Sample-aware visualization mapping:
      - numeric: use linear bbox mapping
      - months/categories: snap x via x_values -> x_ticks_px
    """
    plot_bbox_px = item.get("plot_bbox_px", None)
    if not (isinstance(plot_bbox_px, list) and len(plot_bbox_px) == 4):
        raise ValueError("Missing valid plot_bbox_px")

    x_mode = item.get("x_mode", "numeric")
    x_values = item.get("x_values", None)
    x_ticks_px = item.get("x_ticks_px", None)

    if (
        x_mode in {"months", "categories"}
        and isinstance(x_values, list)
        and isinstance(x_ticks_px, list)
        and len(x_values) > 0
        and len(x_values) == len(x_ticks_px)
    ):
        # map x by nearest allowed x_value, then take corresponding tick pixel
        dists = [abs(float(x) - float(v)) for v in x_values]
        idx = int(np.argmin(dists))
        x_px = float(x_ticks_px[idx])
        y_px = data_y_to_pixel_with_bbox(y, plot_bbox_px, y_min, y_max)
        return x_px, y_px

    # numeric fallback
    return data_to_pixel_numeric_with_bbox(
        x=x,
        y=y,
        plot_bbox_px=plot_bbox_px,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(
    manifest_path: str,
    reconstructed_dir: str,
    line_id: Optional[int],
    x_thresh: float,
    y_thresh: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
):
    items = read_manifest(manifest_path)

    all_tp = 0
    all_fp = 0
    all_fn = 0

    all_x_err = []
    all_y_err = []
    all_euc_err = []

    per_sample = []

    for item in items:
        sample_id = item.get("id")
        gt_csv = item["csv"]
        pred_csv = os.path.join(reconstructed_dir, f"{sample_id}_reconstructed.csv")

        if not os.path.exists(pred_csv):
            print(f"[WARN] Missing reconstructed CSV for sample {sample_id}")
            continue

        gt_points = load_gt_csv(gt_csv, line_id=line_id)
        pred_points, pred_points_px_from_csv = load_reconstructed_csv_with_pixels(pred_csv)

        metrics = greedy_match_data(
            pred_points=pred_points,
            gt_points=gt_points,
            x_thresh=x_thresh,
            y_thresh=y_thresh,
        )

        all_tp += metrics["tp"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]

        if metrics["mean_abs_x_error"] is not None:
            all_x_err.extend([m[3] for m in metrics["matches"]])
        if metrics["mean_abs_y_error"] is not None:
            all_y_err.extend([m[4] for m in metrics["matches"]])
        if metrics["mean_euclidean_error"] is not None:
            all_euc_err.extend([m[2] for m in metrics["matches"]])

        image_path = item.get("full", None)

        # GT overlay pixels: use exact GT points if available, else sample-aware projection
        if manifest_has_points_px(item):
            gt_points_px = load_gt_points_px_from_manifest(item, line_id=line_id)
        else:
            gt_points_px = [
                data_to_pixel_sample_aware(x, y, item, x_min, x_max, y_min, y_max)
                for x, y in gt_points
            ]

        # Prediction overlay pixels:
        # Prefer direct predicted pixels from reconstructed CSV to avoid pixel->data->pixel drift
        if len(pred_points_px_from_csv) == len(pred_points) and len(pred_points_px_from_csv) > 0:
            pred_points_px = pred_points_px_from_csv
        else:
            pred_points_px = [
                data_to_pixel_sample_aware(x, y, item, x_min, x_max, y_min, y_max)
                for x, y in pred_points
            ]

        per_sample.append({
            "id": sample_id,
            "image_path": image_path,
            "plot_bbox_px": item.get("plot_bbox_px", None),
            "gt_points_px": gt_points_px,
            "pred_points_px": pred_points_px,
            "num_gt": len(gt_points),
            "num_pred": len(pred_points),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_abs_x_error": metrics["mean_abs_x_error"],
            "mean_abs_y_error": metrics["mean_abs_y_error"],
            "mean_euclidean_error": metrics["mean_euclidean_error"],
        })

    precision = all_tp / max(1, all_tp + all_fp)
    recall = all_tp / max(1, all_tp + all_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    summary = {
        "num_samples": len(per_sample),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_x_error": float(np.mean(all_x_err)) if all_x_err else None,
        "mean_abs_y_error": float(np.mean(all_y_err)) if all_y_err else None,
        "mean_euclidean_error": float(np.mean(all_euc_err)) if all_euc_err else None,
        "x_thresh": x_thresh,
        "y_thresh": y_thresh,
    }

    return summary, per_sample


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--reconstructed_dir", type=str, required=True)
    ap.add_argument("--line_id", type=int, default=0)

    ap.add_argument("--x_thresh", type=float, default=0.5)
    ap.add_argument("--y_thresh", type=float, default=10.0)

    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    ap.add_argument("--examples_dir", type=str, default="")
    ap.add_argument("--num_best", type=int, default=3)
    ap.add_argument("--num_worst", type=int, default=3)
    ap.add_argument("--num_examples", type=int, default=3)

    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    summary, per_sample = evaluate(
        manifest_path=args.manifest,
        reconstructed_dir=args.reconstructed_dir,
        line_id=args.line_id,
        x_thresh=args.x_thresh,
        y_thresh=args.y_thresh,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.examples_dir:
        save_example_visualizations(
            per_sample=per_sample,
            examples_dir=args.examples_dir,
            num_best=args.num_best,
            num_worst=args.num_worst,
            num_regular=args.num_examples,
        )
        print(f"\nSaved example visualizations to: {args.examples_dir}")

    if args.save_json:
        out = {
            "summary": summary,
            "per_sample": per_sample,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()