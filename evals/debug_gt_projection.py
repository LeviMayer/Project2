import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


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


def read_points(csv_path: str) -> Dict[int, List[Tuple[float, float]]]:
    by_line: Dict[int, List[Tuple[float, float]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (TypeError, ValueError):
                continue

            try:
                line_id = int(row.get("line_id", 0))
            except (TypeError, ValueError):
                line_id = 0

            by_line.setdefault(line_id, []).append((x, y))

    for line_id in by_line:
        by_line[line_id] = sorted(by_line[line_id], key=lambda p: p[0])

    return by_line


def to_pixel_dataset_style(
    x: float,
    y: float,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[int, int]:
    plot_left = int(0.12 * width)
    plot_right = int((0.12 + 0.82) * width)

    plot_bottom = int(0.14 * height)
    plot_top = int((0.14 + 0.78) * height)

    px = plot_left + (x - x_min) / max(x_max - x_min, 1e-8) * (plot_right - plot_left)
    py = plot_bottom + (y - y_min) / max(y_max - y_min, 1e-8) * (plot_top - plot_bottom)

    py = height - py

    px = int(np.clip(round(px), 0, width - 1))
    py = int(np.clip(round(py), 0, height - 1))
    return px, py


def draw_cross(draw: ImageDraw.ImageDraw, x: int, y: int, r: int = 4):
    draw.line((x - r, y, x + r, y), width=2)
    draw.line((x, y - r, x, y + r), width=2)


def visualize_sample(
    image_path: str,
    csv_path: str,
    out_path: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    line_id: int = 0,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    all_points = read_points(csv_path)
    pts = all_points.get(line_id, [])

    pix = [
        to_pixel_dataset_style(
            x=x,
            y=y,
            width=width,
            height=height,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
        for x, y in pts
    ]

    # Draw line in red
    if len(pix) >= 2:
        draw.line(pix, width=3, fill=(255, 0, 0))

    # Draw points in green
    for px, py in pix:
        draw_cross(draw, px, py, r=4)

    # Draw plot bbox in blue for debugging
    plot_left = int(0.12 * width)
    plot_right = int((0.12 + 0.82) * width)
    plot_bottom = int(0.14 * height)
    plot_top = int((0.14 + 0.78) * height)
    bbox = [
        (plot_left, height - plot_top),
        (plot_right, height - plot_top),
        (plot_right, height - plot_bottom),
        (plot_left, height - plot_bottom),
        (plot_left, height - plot_top),
    ]
    draw.line(bbox, width=2, fill=(0, 0, 255))

    image.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--image_key", type=str, default="full")
    ap.add_argument("--line_id", type=int, default=0)

    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    args = ap.parse_args()

    items = read_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end_idx = min(args.start_index + args.num_samples, len(items))

    for i in range(args.start_index, end_idx):
        rec = items[i]
        sample_id = rec.get("id", f"sample_{i:06d}")
        image_path = rec.get(args.image_key)
        csv_path = rec.get("csv")

        if image_path is None or csv_path is None:
            print(f"[WARN] Missing image/csv for {sample_id}")
            continue

        out_path = out_dir / f"{sample_id}_gt_projection.png"

        try:
            visualize_sample(
                image_path=image_path,
                csv_path=csv_path,
                out_path=str(out_path),
                x_min=args.x_min,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=args.y_max,
                line_id=args.line_id,
            )
            print("saved:", out_path)
        except Exception as e:
            print(f"[WARN] Failed on {sample_id}: {e}")


if __name__ == "__main__":
    main()