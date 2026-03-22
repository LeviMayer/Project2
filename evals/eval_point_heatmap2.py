import os
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np


# --------------------------------------------------
# Manifest / CSV
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
    with open(manifest_path, "r") as f:
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


def load_line_csv(csv_path: str, line_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []

    with open(csv_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 3:
                continue

            x_str, y_str, lid_str = parts

            try:
                x = float(x_str)
                y = float(y_str)
                lid = int(lid_str)
            except ValueError:
                continue

            if line_id is not None and lid != int(line_id):
                continue

            xs.append(x)
            ys.append(y)

    if len(xs) == 0:
        raise ValueError(f"No samples found in {csv_path} for line_id={line_id}")

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    idx = np.argsort(xs)
    return xs[idx], ys[idx]


# --------------------------------------------------
# GT loading
# --------------------------------------------------

def manifest_has_points_px(item: Dict) -> bool:
    pts = item.get("points_px", None)
    return isinstance(pts, list) and len(pts) > 0


def load_gt_points_from_manifest(
    item: Dict,
    line_id: Optional[int] = None,
) -> List[Tuple[float, float]]:
    pts = item.get("points_px", [])
    out = []

    for p in pts:
        try:
            lid = int(p.get("line_id", 0))
            x = float(p["x"])
            y = float(p["y"])
        except (KeyError, TypeError, ValueError):
            continue

        if line_id is not None and lid != int(line_id):
            continue

        out.append((x, y))

    out = sorted(out, key=lambda t: t[0])
    return out


# --------------------------------------------------
# Coordinate mapping fallback (older datasets)
# --------------------------------------------------

def data_to_pixel_dataset_style(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    plot_left = int(0.12 * width)
    plot_right = int((0.12 + 0.82) * width)

    plot_bottom = int(0.14 * height)
    plot_top = int((0.14 + 0.78) * height)

    px = plot_left + (xs - x_min) / max(x_max - x_min, 1e-8) * (plot_right - plot_left)
    py = plot_bottom + (ys - y_min) / max(y_max - y_min, 1e-8) * (plot_top - plot_bottom)

    py = height - py

    return list(zip(px.tolist(), py.tolist()))


def load_gt_points(
    item: Dict,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    line_id: Optional[int],
) -> List[Tuple[float, float]]:
    """
    Prefer exact manifest pixel GTs. Fall back to CSV->pixel mapping.
    """
    if manifest_has_points_px(item):
        gt_points = load_gt_points_from_manifest(item, line_id=line_id)
        if len(gt_points) > 0:
            return gt_points

    csv_path = item["csv"]
    xs, ys = load_line_csv(csv_path, line_id=line_id)
    return data_to_pixel_dataset_style(
        xs=xs,
        ys=ys,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


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
    Extract exactly one best peak per slot if its max score >= slot_threshold.

    Returns:
        list of (x, y, score, slot_idx)
    """
    if point_slots.ndim != 3:
        raise ValueError("point_slots must have shape [K,H,W]")

    K, H, W = point_slots.shape
    peaks = []

    usable_k = K if max_slots is None else min(K, max_slots)

    for slot_idx in range(usable_k):
        hm = point_slots[slot_idx]
        flat_idx = int(np.argmax(hm))
        score = float(hm.reshape(-1)[flat_idx])

        if score < slot_threshold:
            continue

        y, x = np.unravel_index(flat_idx, (H, W))
        peaks.append((int(x), int(y), score, slot_idx))

    peaks.sort(key=lambda t: t[3])
    return peaks


# --------------------------------------------------
# Matching
# --------------------------------------------------

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def greedy_match(
    pred_points: List[Tuple[float, float]],
    gt_points: List[Tuple[float, float]],
    dist_thresh: float = 6.0,
):
    pairs = []
    for pi, pp in enumerate(pred_points):
        for gi, gp in enumerate(gt_points):
            d = euclidean(pp, gp)
            if d <= dist_thresh:
                pairs.append((d, pi, gi))

    pairs.sort(key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    matches = []

    for d, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, d))

    tp = len(matches)
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    mean_dist = float(np.mean([m[2] for m in matches])) if matches else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_dist": mean_dist,
        "matches": matches,
    }


# --------------------------------------------------
# Main eval
# --------------------------------------------------

def evaluate(
    manifest_path: str,
    pred_dir: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    slot_threshold: float,
    dist_thresh: float,
    line_id: Optional[int],
    max_slots: Optional[int],
):
    items = read_manifest(manifest_path)

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_match_dists = []
    per_sample = []

    for it in items:
        sample_id = it.get("id")

        pred_path = os.path.join(pred_dir, f"{sample_id}_point_slots.npy")
        if not os.path.exists(pred_path):
            print(f"[WARN] Missing prediction for sample {sample_id}")
            continue

        point_slots = load_point_slots(pred_path)  # [K,H,W]
        _, H, W = point_slots.shape

        gt_points = load_gt_points(
            item=it,
            width=W,
            height=H,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            line_id=line_id,
        )

        pred_peaks = extract_slot_peaks(
            point_slots=point_slots,
            slot_threshold=slot_threshold,
            max_slots=max_slots,
        )
        pred_points = [(x, y) for x, y, _score, _slot_idx in pred_peaks]

        metrics = greedy_match(
            pred_points=pred_points,
            gt_points=gt_points,
            dist_thresh=dist_thresh,
        )

        all_tp += metrics["tp"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]

        if metrics["mean_dist"] is not None:
            all_match_dists.extend([m[2] for m in metrics["matches"]])

        per_sample.append({
            "id": sample_id,
            "num_gt": len(gt_points),
            "num_pred": len(pred_points),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_dist": metrics["mean_dist"],
            "pred_slots": [
                {
                    "slot_idx": int(slot_idx),
                    "x": int(x),
                    "y": int(y),
                    "score": float(score),
                }
                for (x, y, score, slot_idx) in pred_peaks
            ],
        })

    precision = all_tp / max(1, all_tp + all_fp)
    recall = all_tp / max(1, all_tp + all_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    mean_dist = float(np.mean(all_match_dists)) if all_match_dists else None

    summary = {
        "num_samples": len(per_sample),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_dist": mean_dist,
    }

    return summary, per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)

    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    ap.add_argument("--slot_threshold", type=float, default=0.05)
    ap.add_argument("--dist_thresh", type=float, default=10.0)
    ap.add_argument("--line_id", type=int, default=0)
    ap.add_argument("--max_slots", type=int, default=None)

    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    summary, per_sample = evaluate(
        manifest_path=args.manifest,
        pred_dir=args.pred_dir,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        slot_threshold=args.slot_threshold,
        dist_thresh=args.dist_thresh,
        line_id=args.line_id,
        max_slots=args.max_slots,
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.save_json:
        out = {
            "summary": summary,
            "per_sample": per_sample,
        }
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()