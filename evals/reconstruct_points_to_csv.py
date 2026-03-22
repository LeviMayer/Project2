import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# --------------------------------------------------
# Manifest
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


# --------------------------------------------------
# Prediction loading
# --------------------------------------------------

def load_point_slots(path: str) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [K,H,W], got shape={arr.shape} for {path}")
    return arr


# --------------------------------------------------
# Slot peak extraction
# --------------------------------------------------

def extract_slot_peaks(
    point_slots: np.ndarray,
    slot_threshold: float = 0.05,
    max_slots: Optional[int] = None,
) -> List[Tuple[int, int, float, int]]:
    """
    Returns:
        list of (x, y, score, slot_idx)
    """
    if point_slots.ndim != 3:
        raise ValueError("point_slots must have shape [K,H,W]")

    K, H, W = point_slots.shape
    usable_k = K if max_slots is None else min(K, max_slots)

    peaks = []
    for slot_idx in range(usable_k):
        hm = point_slots[slot_idx]
        flat_idx = int(np.argmax(hm))
        score = float(hm.reshape(-1)[flat_idx])

        if score < slot_threshold:
            continue

        y, x = np.unravel_index(flat_idx, (H, W))
        peaks.append((int(x), int(y), score, int(slot_idx)))

    return peaks


# --------------------------------------------------
# Pixel -> data mapping
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
    """
    plot_bbox_px = [x0, y0, w, h] in image coordinates (origin top-left)
    """
    x0, y0, w, h = plot_bbox_px

    x_norm = (px - x0) / max(w, 1e-8)
    y_norm = (py - y0) / max(h, 1e-8)

    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    data_x = x_min + x_norm * (x_max - x_min)
    data_y = y_max - y_norm * (y_max - y_min)

    return float(data_x), float(data_y)


def pixel_to_data_fallback(
    px: float,
    py: float,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float]:
    """
    Fallback using the old approximate dataset box.
    """
    plot_left = int(0.12 * width)
    plot_right = int((0.12 + 0.82) * width)

    plot_bottom = int(0.14 * height)
    plot_top = int((0.14 + 0.78) * height)

    # convert to top-left bbox form
    x0 = float(plot_left)
    y0 = float(height - plot_top)
    w = float(plot_right - plot_left)
    h = float(plot_top - plot_bottom)

    return pixel_to_data_with_bbox(
        px=px,
        py=py,
        plot_bbox_px=[x0, y0, w, h],
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def snap_x_to_ticks(
    x_px: float,
    x_ticks_px: List[float],
    x_values: List[float],
) -> float:
    """
    Snap x to nearest categorical/month tick center.
    """
    if len(x_ticks_px) == 0 or len(x_values) == 0 or len(x_ticks_px) != len(x_values):
        raise ValueError("x_ticks_px and x_values must be non-empty and same length")

    dists = [abs(x_px - tx) for tx in x_ticks_px]
    idx = int(np.argmin(dists))
    return float(x_values[idx])


# --------------------------------------------------
# Reconstruction
# --------------------------------------------------

def reconstruct_sample(
    item: Dict,
    pred_dir: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    slot_threshold: float,
    max_slots: Optional[int],
) -> Dict:
    sample_id = item.get("id")
    pred_path = os.path.join(pred_dir, f"{sample_id}_point_slots.npy")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing prediction: {pred_path}")

    point_slots = load_point_slots(pred_path)   # [K,H,W]
    _, H, W = point_slots.shape

    pred_peaks = extract_slot_peaks(
        point_slots=point_slots,
        slot_threshold=slot_threshold,
        max_slots=max_slots,
    )

    # sort by x for reconstruction order
    pred_peaks = sorted(pred_peaks, key=lambda t: t[0])

    plot_bbox_px = item.get("plot_bbox_px", None)
    use_exact_bbox = (
        isinstance(plot_bbox_px, list)
        and len(plot_bbox_px) == 4
        and all(isinstance(v, (int, float)) for v in plot_bbox_px)
    )

    x_mode = item.get("x_mode", "numeric")
    x_values = item.get("x_values", None)
    x_ticks_px = item.get("x_ticks_px", None)
    x_labels = item.get("x_labels", None)

    use_tick_snap = (
        x_mode in {"months", "categories"}
        and isinstance(x_values, list)
        and isinstance(x_ticks_px, list)
        and len(x_values) > 0
        and len(x_values) == len(x_ticks_px)
    )

    reconstructed = []
    for x_px, y_px, score, slot_idx in pred_peaks:
        # X reconstruction
        if use_tick_snap:
            x_data = snap_x_to_ticks(
                x_px=float(x_px),
                x_ticks_px=[float(v) for v in x_ticks_px],
                x_values=[float(v) for v in x_values],
            )
        else:
            if use_exact_bbox:
                x_data, _ = pixel_to_data_with_bbox(
                    px=x_px,
                    py=y_px,
                    plot_bbox_px=plot_bbox_px,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            else:
                x_data, _ = pixel_to_data_fallback(
                    px=x_px,
                    py=y_px,
                    width=W,
                    height=H,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )

        # Y reconstruction stays continuous
        if use_exact_bbox:
            _, y_data = pixel_to_data_with_bbox(
                px=x_px,
                py=y_px,
                plot_bbox_px=plot_bbox_px,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        else:
            _, y_data = pixel_to_data_fallback(
                px=x_px,
                py=y_px,
                width=W,
                height=H,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )

        reconstructed.append({
            "slot_idx": int(slot_idx),
            "x_px": int(x_px),
            "y_px": int(y_px),
            "score": float(score),
            "x": float(x_data),
            "y": float(y_data),
            "line_id": 0,
        })

    return {
        "id": sample_id,
        "pred_path": pred_path,
        "plot_bbox_px": plot_bbox_px if use_exact_bbox else None,
        "x_mode": x_mode,
        "x_values": x_values,
        "x_labels": x_labels,
        "x_ticks_px": x_ticks_px,
        "used_tick_snap": use_tick_snap,
        "num_points": len(reconstructed),
        "points": reconstructed,
    }


def save_reconstruction_csv(out_csv_path: str, points: List[Dict]) -> None:
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "line_id", "slot_idx", "score", "x_px", "y_px"])
        for p in points:
            w.writerow([
                p["x"],
                p["y"],
                p["line_id"],
                p["slot_idx"],
                p["score"],
                p["x_px"],
                p["y_px"],
            ])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    ap.add_argument("--slot_threshold", type=float, default=0.05)
    ap.add_argument("--max_slots", type=int, default=None)

    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--num_samples", type=int, default=-1)

    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    items = read_manifest(args.manifest)

    start_idx = max(0, args.start_index)
    if args.num_samples is None or args.num_samples < 0:
        end_idx = len(items)
    else:
        end_idx = min(start_idx + args.num_samples, len(items))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    num_ok = 0
    num_missing = 0

    for i in range(start_idx, end_idx):
        item = items[i]
        sample_id = item.get("id", f"sample_{i:06d}")

        try:
            result = reconstruct_sample(
                item=item,
                pred_dir=args.pred_dir,
                x_min=args.x_min,
                x_max=args.x_max,
                y_min=args.y_min,
                y_max=args.y_max,
                slot_threshold=args.slot_threshold,
                max_slots=args.max_slots,
            )

            out_csv_path = out_dir / f"{sample_id}_reconstructed.csv"
            save_reconstruction_csv(str(out_csv_path), result["points"])

            result["out_csv"] = str(out_csv_path)
            all_results.append(result)
            num_ok += 1

            if num_ok % 50 == 0:
                print(f"[INFO] reconstructed {num_ok} samples")

        except FileNotFoundError:
            print(f"[WARN] Missing prediction for sample {sample_id}")
            num_missing += 1
        except Exception as e:
            print(f"[WARN] Failed on sample {sample_id}: {e}")

    summary = {
        "num_requested": end_idx - start_idx,
        "num_reconstructed": num_ok,
        "num_missing_predictions": num_missing,
        "out_dir": str(out_dir),
        "slot_threshold": args.slot_threshold,
        "max_slots": args.max_slots,
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.save_json:
        payload = {
            "summary": summary,
            "samples": all_results,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()