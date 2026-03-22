import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torchvision.transforms as T


class LineHeatmapDataset(Dataset):
    """
    Dataset for line-chart supervision with slot-based point targets.

    Preferred supervision path (for synthetic v3):
        - use exact pixel points from manifest["points_px"]

    Fallback path (older datasets / real data):
        - read CSV and map data coords -> pixel coords via _to_pixel(...)

    Returns:
        image:            [3, H, W]
        line_heatmap:     [1, H, W]
        point_heatmaps:   [K, H, W]
        point_valid_mask: [K]
        num_points:       scalar
        id:               sample id
        plot_bbox_px:     [4] optional metadata tensor
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

        image = Image.open(img_path).convert("RGB")
        image = self.image_tf(image)  # [3,H,W], range [0,1]

        point_records = self._get_point_records(rec)
        if len(point_records) == 0:
            raise ValueError(f"No valid points found for sample {rec.get('id', idx)}")

        line_heatmap = self._build_line_heatmap_from_records(
            point_records=point_records,
            height=self.heatmap_size,
            width=self.heatmap_size,
        )

        point_heatmaps, point_valid_mask, num_points = self._build_point_slot_heatmaps_from_records(
            point_records=point_records,
            height=self.heatmap_size,
            width=self.heatmap_size,
        )

        plot_bbox_px = rec.get("plot_bbox_px", None)
        if plot_bbox_px is None:
            plot_bbox_px = [-1.0, -1.0, -1.0, -1.0]

        return {
            "image": image,
            "line_heatmap": torch.from_numpy(line_heatmap).unsqueeze(0),   # [1,H,W]
            "point_heatmaps": torch.from_numpy(point_heatmaps),            # [K,H,W]
            "point_valid_mask": torch.from_numpy(point_valid_mask),        # [K]
            "num_points": torch.tensor(num_points, dtype=torch.long),
            "plot_bbox_px": torch.tensor(plot_bbox_px, dtype=torch.float32),
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

    def _to_pixel(self, x: float, y: float, width: int, height: int) -> Tuple[int, int]:
        """
        Fallback mapping for datasets without exact pixel metadata.
        """
        plot_left = int(0.12 * width)
        plot_right = int((0.12 + 0.82) * width)

        plot_bottom = int(0.14 * height)
        plot_top = int((0.14 + 0.78) * height)

        px = plot_left + (x - self.x_min) / max(self.x_max - self.x_min, 1e-8) * (plot_right - plot_left)
        py = plot_bottom + (y - self.y_min) / max(self.y_max - self.y_min, 1e-8) * (plot_top - plot_bottom)

        py = height - py

        px = int(np.clip(round(px), 0, width - 1))
        py = int(np.clip(round(py), 0, height - 1))

        return px, py

    def _manifest_has_points_px(self, rec: dict) -> bool:
        pts = rec.get("points_px", None)
        return isinstance(pts, list) and len(pts) > 0

    def _get_point_records_from_manifest(
        self,
        rec: dict,
        width: int,
        height: int,
    ) -> List[Dict[str, int]]:
        """
        Use exact plotted pixel points stored in manifest.
        Each item returned:
            {"line_id": int, "px": int, "py": int}
        """
        pts = rec.get("points_px", [])
        out: List[Dict[str, int]] = []

        for p in pts:
            try:
                line_id = int(p.get("line_id", 0))
                px = int(np.clip(round(float(p["x"])), 0, width - 1))
                py = int(np.clip(round(float(p["y"])), 0, height - 1))
            except (KeyError, TypeError, ValueError):
                continue

            out.append({
                "line_id": line_id,
                "px": px,
                "py": py,
            })

        # sort globally by x, then by line_id for deterministic ordering
        out = sorted(out, key=lambda d: (d["px"], d["line_id"], d["py"]))
        return out

    def _get_point_records_from_csv(
        self,
        rec: dict,
        width: int,
        height: int,
    ) -> List[Dict[str, int]]:
        """
        Fallback for older datasets:
            CSV -> numeric points -> approximate pixel projection
        """
        csv_rel = rec["csv"]
        csv_path = self.root_dir / csv_rel

        line_points = self._read_points(csv_path)
        if not line_points or sum(len(v) for v in line_points.values()) == 0:
            return []

        out: List[Dict[str, int]] = []
        for line_id, pts in line_points.items():
            for x, y in pts:
                px, py = self._to_pixel(x, y, width, height)
                out.append({
                    "line_id": int(line_id),
                    "px": int(px),
                    "py": int(py),
                })

        out = sorted(out, key=lambda d: (d["px"], d["line_id"], d["py"]))
        return out

    def _get_point_records(
        self,
        rec: dict,
    ) -> List[Dict[str, int]]:
        """
        Main entry:
        Prefer exact manifest pixel points, fallback to CSV mapping.
        """
        width = self.heatmap_size
        height = self.heatmap_size

        if self._manifest_has_points_px(rec):
            pts = self._get_point_records_from_manifest(rec, width=width, height=height)
            if len(pts) > 0:
                return pts

        return self._get_point_records_from_csv(rec, width=width, height=height)

    def _group_point_records_by_line(
        self,
        point_records: List[Dict[str, int]],
    ) -> Dict[int, List[Tuple[int, int]]]:
        by_line: Dict[int, List[Tuple[int, int]]] = {}
        for p in point_records:
            line_id = int(p["line_id"])
            by_line.setdefault(line_id, []).append((int(p["px"]), int(p["py"])))

        for line_id in by_line:
            by_line[line_id] = sorted(by_line[line_id], key=lambda t: t[0])

        return by_line

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

    def _has_valid_points(self, rec: dict) -> bool:
        if self._manifest_has_points_px(rec):
            pts = rec.get("points_px", [])
            return len(pts) > 0

        csv_rel = rec.get("csv")
        if csv_rel is None:
            return False

        csv_path = self.root_dir / csv_rel
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
            if self._has_valid_points(rec):
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

    def _build_line_heatmap_from_records(
        self,
        point_records: List[Dict[str, int]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """
        Build line heatmap from exact pixel points or fallback-projected points.
        """
        canvas = np.zeros((height, width), dtype=np.float32)

        by_line = self._group_point_records_by_line(point_records)
        for _line_id, pix in by_line.items():
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
        canvas = np.zeros((height, width), dtype=np.float32)
        canvas[py, px] = 1.0
        heatmap = self._gaussian_blur(canvas, self.point_sigma)
        return heatmap.astype(np.float32)

    def _build_point_slot_heatmaps_from_records(
        self,
        point_records: List[Dict[str, int]],
        height: int,
        width: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build slot-based point heatmaps from pixel records.

        Returns:
            point_heatmaps: [K,H,W]
            point_valid_mask: [K]
            num_points: int
        """
        if len(point_records) == 0:
            point_heatmaps = np.zeros((self.max_points, height, width), dtype=np.float32)
            point_valid_mask = np.zeros((self.max_points,), dtype=np.float32)
            return point_heatmaps, point_valid_mask, 0

        point_records = sorted(point_records, key=lambda d: (d["px"], d["line_id"], d["py"]))
        point_records = point_records[:self.max_points]
        num_points = len(point_records)

        point_heatmaps = np.zeros((self.max_points, height, width), dtype=np.float32)
        point_valid_mask = np.zeros((self.max_points,), dtype=np.float32)

        for slot_idx, p in enumerate(point_records):
            px = int(p["px"])
            py = int(p["py"])
            point_heatmaps[slot_idx] = self._build_single_point_heatmap(
                px=px,
                py=py,
                height=height,
                width=width,
            )
            point_valid_mask[slot_idx] = 1.0

        return point_heatmaps.astype(np.float32), point_valid_mask.astype(np.float32), num_points