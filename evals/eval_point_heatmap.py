import os
import json
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter


# --------------------------------------------------
# Reuse / adapt from your repo
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


def load_line_csv(csv_path: str, line_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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
            if int(lid_str) != int(line_id):
                continue
            xs.append(float(x_str))
            ys.append(float(y_str))

    if len(xs) == 0:
        raise ValueError(f"No samples found for line_id={line_id} in {csv_path}")

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    idx = np.argsort(xs)
    return xs[idx], ys[idx]


# --------------------------------------------------
# Coordinate mapping
# --------------------------------------------------

def data_to_pixel(
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> List[Tuple[float, float]]:
    # x increases left -> right
    # y increases bottom -> top in data, but image y is top -> bottom
    x_norm = (xs - x_min) / max(1e-8, (x_max - x_min))
    y_norm = (ys - y_min) / max(1e-8, (y_max - y_min))

    px = x_norm * (width - 1)
    py = (1.0 - y_norm) * (height - 1)

    return list(zip(px.tolist(), py.tolist()))


# --------------------------------------------------
# Heatmap loading
# --------------------------------------------------

def load_heatmap(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        hm = np.load(path)
    else:
        img = Image.open(path).convert("L")
        hm = np.asarray(img, dtype=np.float32) / 255.0

    hm = hm.astype(np.float32)

    if hm.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape={hm.shape} for {path}")

    return hm


# --------------------------------------------------
# Peak extraction
# --------------------------------------------------

def extract_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    nms_kernel: int = 5,
    max_peaks: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """
    Returns peaks as (x, y, score)
    """
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be HxW")

    max_map = maximum_filter(heatmap, size=nms_kernel, mode="constant")
    keep = (heatmap == max_map) & (heatmap >= threshold)

    ys, xs = np.where(keep)
    scores = heatmap[ys, xs]

    peaks = [(int(x), int(y), float(s)) for x, y, s in zip(xs, ys, scores)]
    peaks.sort(key=lambda t: t[2], reverse=True)

    if max_peaks is not None:
        peaks = peaks[:max_peaks]

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
    """
    Greedy bipartite-style matching by smallest distance.
    """
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
    line_id: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    threshold: float,
    nms_kernel: int,
    dist_thresh: float,
    max_peaks: Optional[int],
):
    items = read_manifest(manifest_path)

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_match_dists = []
    per_sample = []

    for it in items:
        sample_id = it.get("id")
        csv_path = it["csv"]

        pred_path_npy = os.path.join(pred_dir, f"{sample_id}_point.npy")
        pred_path_png = os.path.join(pred_dir, f"{sample_id}_point.png")

        if os.path.exists(pred_path_npy):
            pred_path = pred_path_npy
        elif os.path.exists(pred_path_png):
            pred_path = pred_path_png
        else:
            print(f"[WARN] Missing prediction for sample {sample_id}")
            continue

        hm = load_heatmap(pred_path)
        H, W = hm.shape

        xs, ys = load_line_csv(csv_path, line_id=line_id)
        gt_points = data_to_pixel(
            xs, ys,
            width=W, height=H,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max
        )

        peaks = extract_peaks(
            hm,
            threshold=threshold,
            nms_kernel=nms_kernel,
            max_peaks=max_peaks,
        )
        pred_points = [(x, y) for x, y, _ in peaks]

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
    ap.add_argument("--line_id", type=int, default=0)

    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)

    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--nms_kernel", type=int, default=5)
    ap.add_argument("--dist_thresh", type=float, default=6.0)
    ap.add_argument("--max_peaks", type=int, default=None)

    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    summary, per_sample = evaluate(
        manifest_path=args.manifest,
        pred_dir=args.pred_dir,
        line_id=args.line_id,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        threshold=args.threshold,
        nms_kernel=args.nms_kernel,
        dist_thresh=args.dist_thresh,
        max_peaks=args.max_peaks,
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