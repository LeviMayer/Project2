import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


class HeatmapCurveExtractor:
    """
    Convert a predicted line heatmap into a discrete curve by taking the
    peak response per x-column.

    Assumptions:
      - Heatmap shape is [H, W] or image file convertible to grayscale
      - Brighter values indicate higher confidence for the line
      - Output is one y-value per x-column
    """

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 100.0,
        smooth_window: int = 5,
        threshold: float = 0.0,
    ) -> None:
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.smooth_window = int(max(1, smooth_window))
        self.threshold = float(threshold)

    def load_heatmap(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def extract_pixels(self, heatmap: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Returns a list of (x_px, y_px, confidence), one point per x-column.
        y_px is in image coordinates (0 at top).
        """
        H, W = heatmap.shape
        points: List[Tuple[int, int, float]] = []

        for x in range(W):
            column = heatmap[:, x]
            y = int(np.argmax(column))
            conf = float(column[y])
            if conf >= self.threshold:
                points.append((x, y, conf))

        return points

    def smooth_y(self, ys: np.ndarray) -> np.ndarray:
        if self.smooth_window <= 1:
            return ys
        k = self.smooth_window
        pad = k // 2
        padded = np.pad(ys, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float32) / float(k)
        out = np.convolve(padded, kernel, mode="valid")
        return out.astype(np.float32)

    def pixel_to_chart(self, xs_px: np.ndarray, ys_px: np.ndarray, W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel coordinates into chart coordinates using the known
        synthetic generator ranges.
        """
        xs = self.x_min + (xs_px / max(W - 1, 1)) * (self.x_max - self.x_min)

        # invert y because image y grows downward, chart y upward
        ys_norm = 1.0 - (ys_px / max(H - 1, 1))
        ys = self.y_min + ys_norm * (self.y_max - self.y_min)
        return xs.astype(np.float32), ys.astype(np.float32)

    def extract_chart_curve(self, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = heatmap.shape
        px_points = self.extract_pixels(heatmap)
        if len(px_points) == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        xs_px = np.asarray([p[0] for p in px_points], dtype=np.float32)
        ys_px = np.asarray([p[1] for p in px_points], dtype=np.float32)
        conf = np.asarray([p[2] for p in px_points], dtype=np.float32)

        ys_px = self.smooth_y(ys_px)
        xs, ys = self.pixel_to_chart(xs_px, ys_px, W=W, H=H)
        return xs, ys, conf

    def save_csv(self, xs: np.ndarray, ys: np.ndarray, out_csv: str) -> None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for x, y in zip(xs, ys):
                writer.writerow([float(x), float(y)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatmap", type=str, required=True, help="Path to grayscale heatmap image")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=10.0)
    parser.add_argument("--y_min", type=float, default=0.0)
    parser.add_argument("--y_max", type=float, default=100.0)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    extractor = HeatmapCurveExtractor(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        smooth_window=args.smooth_window,
        threshold=args.threshold,
    )

    heatmap = extractor.load_heatmap(args.heatmap)
    xs, ys, _conf = extractor.extract_chart_curve(heatmap)
    extractor.save_csv(xs, ys, args.out_csv)

    print(f"[INFO] extracted {len(xs)} points")
    print(f"[INFO] wrote CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
