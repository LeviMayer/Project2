import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torchvision.transforms as T


class LineHeatmapDataset(Dataset):
    """
    Dataset for line-chart supervision with slot-based point targets.

    Returns:
        image:            [3, H, W]
        line_heatmap:     [1, H, W]   -> full curve heatmap
        point_heatmaps:   [K, H, W]   -> one channel per sorted point slot
        point_valid_mask: [K]         -> 1 if slot is used, else 0
        num_points:       scalar       -> number of valid points used
        id:               sample id
    """

    def __init__(
        self,
        root_dir: str,
        manifest_path: str,
        image_key: str = "full",
        image_size: int = 224,
        heatmap_size: int = 224,
        sigma: float = 2.5,
        point_sigma: float = 0.6,
        max_points: int = 16,
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
        self.records = self._filter_valid_records(self.records)
        print(f"[INFO] valid samples: {len(self.records)}")
        self.image_key = image_key
        self.image_size = int(image_size)
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.point_sigma = float(point_sigma)
        self.max_points = int(max_points)

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        self.image_tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),
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
        if not line_points or sum(len(v) for v in line_points.values()) == 0:
            raise ValueError(f"No valid numeric points found in {csv_path}")

        line_heatmap = self._build_line_heatmap(
            line_points=line_points,
            height=self.heatmap_size,
            width=self.heatmap_size,
        )

        point_heatmaps, point_valid_mask, num_points = self._build_point_slot_heatmaps(
            line_points=line_points,
            height=self.heatmap_size,
            width=self.heatmap_size,
        )

        return {
            "image": image,
            "line_heatmap": torch.from_numpy(line_heatmap).unsqueeze(0),          # [1,H,W]
            "point_heatmaps": torch.from_numpy(point_heatmaps),                   # [K,H,W]
            "point_valid_mask": torch.from_numpy(point_valid_mask),               # [K]
            "num_points": torch.tensor(num_points, dtype=torch.long),
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
            dict: line_id -> list[(x, y)] sorted by x
        """
        by_line: Dict[int, List[Tuple[float, float]]] = {}
        with csv_path.open("r", encoding="utf-8") as f:
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

    def _flatten_and_sort_points(
        self,
        line_points: Dict[int, List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        """
        Flatten all lines into a single list of points and sort globally by x.
        For now this is fine for single-line charts and a pragmatic start for Phase 2.
        """
        all_points: List[Tuple[float, float]] = []
        for _line_id, pts in line_points.items():
            all_points.extend(pts)

        all_points = sorted(all_points, key=lambda p: p[0])
        return all_points

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

    def _has_valid_points(self, csv_path: Path) -> bool:
        try:
            by_line = self._read_points(csv_path)
            total_points = sum(len(v) for v in by_line.values())
            return total_points > 0
        except Exception:
            return False


    def _filter_valid_records(self, records: List[dict]) -> List[dict]:
        valid_records = []
        dropped = 0

        for rec in records:
            csv_rel = rec.get("csv")
            if csv_rel is None:
                dropped += 1
                continue

            csv_path = self.root_dir / csv_rel
            if self._has_valid_points(csv_path):
                valid_records.append(rec)
            else:
                dropped += 1

        print(f"[INFO] filtered invalid samples: {dropped}")
        return valid_records

    def _gaussian_blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return img.astype(np.float32)

        out = gaussian_filter(img.astype(np.float32), sigma=sigma)
        out = out / max(out.max(), 1e-8)
        return out.astype(np.float32)

    def _build_line_heatmap(
        self,
        line_points: Dict[int, List[Tuple[float, float]]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Heatmap for the whole line / curve.
        """
        canvas = np.zeros((height, width), dtype=np.float32)

        for _line_id, pts in line_points.items():
            pix = [self._to_pixel(x, y, width, height) for x, y in pts]
            self._draw_line_segments(canvas, pix)

        heatmap = self._gaussian_blur(canvas, self.sigma)
        return heatmap.astype(np.float32)

    def _build_single_point_heatmap(
        self,
        px: int,
        py: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Build one heatmap channel for exactly one support point.
        """
        canvas = np.zeros((height, width), dtype=np.float32)
        canvas[py, px] = 1.0
        heatmap = self._gaussian_blur(canvas, self.point_sigma)
        return heatmap.astype(np.float32)

    def _build_point_slot_heatmaps(
        self,
        line_points: Dict[int, List[Tuple[float, float]]],
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build slot-based point heatmaps.

        Returns:
            point_heatmaps: [K,H,W]
            point_valid_mask: [K]
            num_points: int
        """
        all_points = self._flatten_and_sort_points(line_points)
        if len(all_points) == 0:
            point_heatmaps = np.zeros((self.max_points, height, width), dtype=np.float32)
            point_valid_mask = np.zeros((self.max_points,), dtype=np.float32)
            return point_heatmaps, point_valid_mask, 0

        all_points = all_points[:self.max_points]
        num_points = len(all_points)

        point_heatmaps = np.zeros((self.max_points, height, width), dtype=np.float32)
        point_valid_mask = np.zeros((self.max_points,), dtype=np.float32)

        for slot_idx, (x, y) in enumerate(all_points):
            px, py = self._to_pixel(x, y, width, height)
            point_heatmaps[slot_idx] = self._build_single_point_heatmap(
                px=px,
                py=py,
                height=height,
                width=width,
            )
            point_valid_mask[slot_idx] = 1.0

        return point_heatmaps.astype(np.float32), point_valid_mask.astype(np.float32), num_points