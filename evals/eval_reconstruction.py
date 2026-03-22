import os
import json
import math
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np


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
    """
    Match points in data space with thresholds on x and y.
    Among allowed matches, use smallest Euclidean distance.
    """
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
# Evaluation
# --------------------------------------------------

def evaluate(
    manifest_path: str,
    reconstructed_dir: str,
    line_id: Optional[int],
    x_thresh: float,
    y_thresh: float,
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
        pred_points = load_reconstructed_csv(pred_csv)

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

    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    summary, per_sample = evaluate(
        manifest_path=args.manifest,
        reconstructed_dir=args.reconstructed_dir,
        line_id=args.line_id,
        x_thresh=args.x_thresh,
        y_thresh=args.y_thresh,
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

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