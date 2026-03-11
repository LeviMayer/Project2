import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class LineHeatmapDataset(Dataset):
    """
    Dataset for synthetic line-chart heatmap supervision.

    Expected manifest format (jsonl, one object per line):
      {
        "id": "chart_000001",
        "full": "images/chart_000001_full.png",
        "masked": "images/chart_000001_masked.png",
        "csv": "labels/chart_000001.csv",
        "n_lines": 1,
        "n_points": 42
      }

    Expected CSV format:
      x,y,line_id
      0.0,23.1,0
      0.5,24.7,0
      ...

    For now this loader builds a SINGLE heatmap channel by drawing all points/segments
    for all lines into one map. That is perfect for the first 1-line prototype and still
    works later for multi-line experiments if you want a merged line heatmap.
    """

    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        image_key: str = "full",
        image_size: int = 224,
        heatmap_size: int = 224,
        sigma: float = 2.5,
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 100.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.root_dir / self.manifest_path

        self.records = self._load_manifest(self.manifest_path)
        self.image_key = image_key
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.image_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        img_rel = rec.get(self.image_key)
        if img_rel is None:
            raise ValueError(f"Record {rec.get('id', idx)} has no image key '{self.image_key}'")
        img_path = self.root_dir / img_rel

        csv_rel = rec["csv"]
        csv_path = self.root_dir / csv_rel

        image = Image.open(img_path).convert("RGB")
        image = self.image_tf(image)  # [3,H,W], range [0,1]

        line_points = self._read_points(csv_path)
        heatmap = self._build_heatmap(line_points, self.heatmap_size, self.heatmap_size)

        return {
            "image": image,
            "line_heatmap": torch.from_numpy(line_heatmap).unsqueeze(0),
            "point_heatmap": torch.from_numpy(point_heatmap).unsqueeze(0),
            "id": rec.get("id", f"sample_{idx:06d}"),
        }

    @staticmethod
    def _load_manifest(path: Path) -> List[dict]:
        records: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    @staticmethod
    def _read_points(csv_path: Path) -> Dict[int, List[Tuple[float, float]]]:
        """
        Returns:
            dict: line_id -> list[(x, y)]
        """
        by_line: Dict[int, List[Tuple[float, float]]] = {}
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row["x"])
                y = float(row["y"])
                line_id = int(row.get("line_id", 0))
                by_line.setdefault(line_id, []).append((x, y))

        # ensure each line is sorted by x
        for line_id in by_line:
            by_line[line_id] = sorted(by_line[line_id], key=lambda p: p[0])
        return by_line

    def _to_pixel(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        """
        Maps chart coordinates into heatmap pixel coordinates.
        x grows left->right, y grows bottom->top in chart coordinates.
        image coordinates use top->bottom, so y is inverted.
        """
        plot_left = int(0.12 * width)
        plot_right = int((0.12 + 0.82) * width)

        plot_bottom = int(0.14 * height)
        plot_top = int((0.14 + 0.78) * height)

        px = plot_left + (x - self.x_min) / max(self.x_max - self.x_min, 1e-8) * (plot_right - plot_left)
        py = plot_bottom + (y - self.y_min) / max(self.y_max - self.y_min, 1e-8) * (plot_top - plot_bottom)

        # invert y for image coordinates
        py = height - py

        px = int(np.clip(round(px), 0, width - 1))
        py = int(np.clip(round(py), 0, height - 1))

        return px, py

    def _draw_line_segments(
        self,
        canvas: np.ndarray,
        pts: List[Tuple[int, int]],
    ) -> None:
        """
        Rasterize line by linear interpolation between consecutive points.
        This avoids sparse labels when only CSV anchor points are drawn.
        """
        if len(pts) == 0:
            return
        if len(pts) == 1:
            x, y = pts[0]
            canvas[y, x] = 1.0
            return

        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, n)
            ys = np.linspace(y0, y1, n)
            xs = np.clip(np.round(xs).astype(np.int32), 0, canvas.shape[1] - 1)
            ys = np.clip(np.round(ys).astype(np.int32), 0, canvas.shape[0] - 1)
            canvas[ys, xs] = 1.0

    def _gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return img

        radius = max(1, int(3 * sigma))
        size = 2 * radius + 1
        x = np.arange(size) - radius
        kernel_1d = np.exp(-(x ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # separable conv, numpy-only
        padded = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
        tmp = np.zeros_like(img, dtype=np.float32)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                tmp[r, c] = np.sum(padded[r, c:c + size] * kernel_1d)

        padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
        out = np.zeros_like(tmp, dtype=np.float32)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                out[r, c] = np.sum(padded[r:r + size, c] * kernel_1d)

        # normalize into [0,1]
        out = out / max(out.max(), 1e-8)
        return out.astype(np.float32)

    def _build_heatmap(
        self,
        line_points: Dict[int, List[Tuple[float, float]]],
        height: int,
        width: int,
    ) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.float32)

        for _line_id, pts in line_points.items():
            pix = [self._to_pixel(x, y, width, height) for x, y in pts]
            self._draw_line_segments(canvas, pix)

        heatmap = self._gaussian_blur(canvas, self.sigma)
        return heatmap.astype(np.float32)
