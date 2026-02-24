import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class LineExDataset(Dataset):
    """
    Erwartet root/out_lineex Struktur aus deinem Generator:
      root/
        images/*.png
        labels/*.csv
        manifest.jsonl
    CSV: x,y,line_id
    """
    def __init__(self, root, manifest="manifest.jsonl", T=64, Lmax=3, use_masked=False):
        self.root = root
        self.T = T
        self.Lmax = Lmax
        self.use_masked = use_masked

        man_path = os.path.join(root, manifest)
        self.items = []
        with open(man_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def _load_csv_lines(self, csv_path):
        arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        lines = {}
        for x, y, lid in arr:
            lid = int(lid)
            lines.setdefault(lid, [[], []])
            lines[lid][0].append(float(x))
            lines[lid][1].append(float(y))

        out = {}
        for lid, (xs, ys) in lines.items():
            xs = np.array(xs, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)
            idx = np.argsort(xs)
            out[lid] = (xs[idx], ys[idx])
        return out

    def __getitem__(self, i):
        it = self.items[i]
        img_rel = it["masked"] if (self.use_masked and it.get("masked")) else it["full"]
        img_path = os.path.join(self.root, img_rel)
        csv_path = os.path.join(self.root, it["csv"])

        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [3,H,W]

        lines = self._load_csv_lines(csv_path)

        y_tgt = np.zeros((self.Lmax, self.T), dtype=np.float32)
        lmask = np.zeros((self.Lmax,), dtype=np.float32)

        lids = sorted(lines.keys())[:self.Lmax]
        for j, lid in enumerate(lids):
            xs, ys = lines[lid]
            xmin, xmax = float(xs.min()), float(xs.max())
            xq = np.linspace(xmin, xmax, self.T).astype(np.float32)
            yq = np.interp(xq, xs, ys).astype(np.float32)

            y_tgt[j] = yq
            lmask[j] = 1.0

        return {
            "image": img,
            "y": torch.from_numpy(y_tgt),          # [Lmax,T]
            "line_mask": torch.from_numpy(lmask),  # [Lmax]
        }
