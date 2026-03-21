import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_curve_csv(csv_path: Path, x_col: str = "x", y_col: str = "y") -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))

    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)

    if len(xs_arr) == 0:
        return xs_arr, ys_arr

    order = np.argsort(xs_arr)
    return xs_arr[order], ys_arr[order]


def load_gt_curve_csv(csv_path: Path, line_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground-truth CSV from generator format:
      x,y,line_id
    """
    xs: List[float] = []
    ys: List[float] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            li = int(row.get("line_id", 0))
            if li != line_id:
                continue
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))

    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)

    if len(xs_arr) == 0:
        return xs_arr, ys_arr

    order = np.argsort(xs_arr)
    return xs_arr[order], ys_arr[order]


def interpolate_curve(xs: np.ndarray, ys: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return np.full_like(x_grid, np.nan, dtype=np.float32)

    # remove duplicate x values if any
    uniq_xs, uniq_idx = np.unique(xs, return_index=True)
    uniq_ys = ys[uniq_idx]

    if len(uniq_xs) == 1:
        return np.full_like(x_grid, uniq_ys[0], dtype=np.float32)

    return np.interp(x_grid, uniq_xs, uniq_ys).astype(np.float32)


def curve_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan")}

    diff = y_true[valid] - y_pred[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_single_sample(
    gt_csv: Path,
    pred_csv: Path,
    line_id: int,
    x_min: float,
    x_max: float,
    n_eval_points: int,
) -> Dict[str, float]:
    gt_x, gt_y = load_gt_curve_csv(gt_csv, line_id=line_id)
    pred_x, pred_y = load_curve_csv(pred_csv, x_col="x", y_col="y")

    x_grid = np.linspace(x_min, x_max, n_eval_points, dtype=np.float32)
    gt_interp = interpolate_curve(gt_x, gt_y, x_grid)
    pred_interp = interpolate_curve(pred_x, pred_y, x_grid)

    metrics = curve_metrics(gt_interp, pred_interp)
    metrics["n_gt_points"] = int(len(gt_x))
    metrics["n_pred_points"] = int(len(pred_x))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True, help="Root dir of original synthetic dataset")
    parser.add_argument("--gt_manifest", type=str, required=True, help="Manifest jsonl used for evaluation split")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted csv files")
    parser.add_argument("--line_id", type=int, default=0, help="GT line_id to evaluate")
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=10.0)
    parser.add_argument("--n_eval_points", type=int, default=256)
    parser.add_argument("--out_json", type=str, default="", help="Optional path to save summary json")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    gt_manifest = Path(args.gt_manifest)
    if not gt_manifest.is_absolute():
        gt_manifest = gt_root / gt_manifest

    pred_dir = Path(args.pred_dir)

    if not gt_manifest.exists():
        raise FileNotFoundError(f"GT manifest not found: {gt_manifest}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")

    records = []
    with gt_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = []
    missing_preds = []

    for rec in records:
        sample_id = rec["id"]
        gt_csv = gt_root / rec["csv"]
        pred_csv = pred_dir / f"{sample_id}.csv"

        if not pred_csv.exists():
            missing_preds.append(sample_id)
            continue

        metrics = evaluate_single_sample(
            gt_csv=gt_csv,
            pred_csv=pred_csv,
            line_id=args.line_id,
            x_min=args.x_min,
            x_max=args.x_max,
            n_eval_points=args.n_eval_points,
        )

        row = {
            "id": sample_id,
            "gt_csv": str(gt_csv),
            "pred_csv": str(pred_csv),
            **metrics,
        }
        results.append(row)

    maes = [r["mae"] for r in results if np.isfinite(r["mae"])]
    rmses = [r["rmse"] for r in results if np.isfinite(r["rmse"])]

    summary = {
        "num_records_in_manifest": len(records),
        "num_evaluated": len(results),
        "num_missing_predictions": len(missing_preds),
        "missing_prediction_ids": missing_preds[:50],
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "median_mae": float(np.median(maes)) if maes else float("nan"),
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "median_rmse": float(np.median(rmses)) if rmses else float("nan"),
        "per_sample": results,
    }

    print("========== Curve Evaluation ==========")
    print(f"Records in manifest : {summary['num_records_in_manifest']}")
    print(f"Evaluated           : {summary['num_evaluated']}")
    print(f"Missing predictions : {summary['num_missing_predictions']}")
    print(f"Mean MAE            : {summary['mean_mae']:.6f}")
    print(f"Median MAE          : {summary['median_mae']:.6f}")
    print(f"Mean RMSE           : {summary['mean_rmse']:.6f}")
    print(f"Median RMSE         : {summary['median_rmse']:.6f}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Wrote summary json to: {out_json}")


if __name__ == "__main__":
    main()