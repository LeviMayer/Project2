#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# ----------------------------
# Optional wandb
# ----------------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ----------------------------
# JEPA imports (repo-local)
# ----------------------------
# This assumes you run from repo root: python train_valuehead.py ...
from app.vjepa.utils import init_video_model


# ----------------------------
# Helpers
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_manifest(manifest_path: str, out_dir: str):
    """
    Reads manifest.jsonl and returns items with absolute paths.
    IMPORTANT: paths in manifest are relative to the manifest directory (dataset root),
    not to the valuehead out_dir.
    Also normalizes Windows backslashes.
    """
    data_root = os.path.dirname(os.path.abspath(manifest_path))

    def _fix(p):
        if p is None:
            return None
        # normalize windows slashes from manifest
        p = p.replace("\\", "/")
        # if relative -> resolve w.r.t. manifest directory
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        # normalize again
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
    # expects header: x,y,line_id
    xs, ys = [], []
    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            x_str, y_str, lid_str = line.split(",")
            if int(lid_str) != int(line_id):
                continue
            xs.append(float(x_str))
            ys.append(float(y_str))
    if len(xs) == 0:
        raise ValueError(f"No samples found for line_id={line_id} in {csv_path}")
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    # sort by x
    idx = np.argsort(xs)
    return xs[idx], ys[idx]


def make_target_vector(
    xs: np.ndarray,
    ys: np.ndarray,
    x_min: float,
    x_max: float,
    K: int,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    normalize_y_to_0_1: bool = False,
) -> np.ndarray:
    """
    Interpolate y on K uniform x positions between [x_min, x_max].
    """
    xq = np.linspace(x_min, x_max, K, dtype=np.float32)
    yq = np.interp(xq, xs, ys).astype(np.float32)

    if normalize_y_to_0_1:
        if y_min is None or y_max is None:
            # fallback: min/max from curve
            lo = float(np.min(ys))
            hi = float(np.max(ys))
        else:
            lo = float(y_min)
            hi = float(y_max)
        denom = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        yq = (yq - lo) / denom
        yq = np.clip(yq, 0.0, 1.0)

    return yq


# ----------------------------
# Dataset
# ----------------------------
class LineExValueDataset(Dataset):
    def __init__(
        self,
        out_dir: str,
        manifest_path: str,
        img_size: int = 224,
        K: int = 64,
        line_id: int = 0,
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: Optional[float] = 0.0,
        y_max: Optional[float] = 100.0,
        normalize_y_to_0_1: bool = False,
    ):
        self.out_dir = out_dir
        self.items = read_manifest(manifest_path, out_dir=out_dir)
        self.img_size = img_size
        self.K = K
        self.line_id = line_id
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.normalize_y_to_0_1 = normalize_y_to_0_1

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = Image.open(it["full"]).convert("RGB")
        x = self.tf(img)  # [3,H,W]

        xs, ys = load_line_csv(it["csv"], line_id=self.line_id)
        tgt = make_target_vector(
            xs, ys,
            x_min=self.x_min, x_max=self.x_max, K=self.K,
            y_min=self.y_min, y_max=self.y_max,
            normalize_y_to_0_1=self.normalize_y_to_0_1
        )
        tgt = torch.from_numpy(tgt)  # [K]
        return x, tgt, it["id"]


# ----------------------------
# ValueHead
# ----------------------------
class ValueHead(nn.Module):
    """
    Pool encoder tokens -> predict K y-values (regression).
    """
    def __init__(self, embed_dim: int, K: int, pool: str = "mean", dropout: float = 0.0):
        super().__init__()
        assert pool in ("mean", "cls"), "pool must be 'mean' or 'cls'"
        self.pool = pool
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(embed_dim, K)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, D]
        if self.pool == "cls":
            pooled = tokens[:, 0]  # assume cls at 0
        else:
            pooled = tokens.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)  # [B,K]


# ----------------------------
# Optional: load JEPA encoder weights
# ----------------------------
def try_load_encoder_from_checkpoint(encoder: nn.Module, ckpt_path: str) -> bool:
    """
    Loads encoder weights from a JEPA checkpoint created by app/vjepa/train.py (common format).
    Returns True if loaded something, else False.
    """
    if not ckpt_path:
        return False
    if not os.path.exists(ckpt_path):
        print(f"[WARN] checkpoint not found: {ckpt_path}")
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Common formats:
    # - {'encoder': state_dict, ...}
    # - {'state_dict': ...}
    # - sometimes nested keys
    state = None
    if isinstance(ckpt, dict):
        if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
            state = ckpt["encoder"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]

    if state is None:
        print(f"[WARN] checkpoint format not recognized, keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        return False

    msg = encoder.load_state_dict(state, strict=False)
    print(f"[INFO] loaded encoder weights from {ckpt_path} with msg: {msg}")
    return True


# ----------------------------
# Training
# ----------------------------
@torch.no_grad()
def make_wandb_preview(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    step: int,
    title: str = "pred_vs_target"
):
    """
    Create a small plot (CPU) and return wandb.Image if available.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().float().cpu().numpy()
    tgt_np = tgt.detach().float().cpu().numpy()

    fig = plt.figure(figsize=(5, 3), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tgt_np, label="target")
    ax.plot(pred_np, label="pred")
    ax.set_title(f"{title} (step={step})")
    ax.legend()
    fig.tight_layout()

    if WANDB_AVAILABLE:
        img = wandb.Image(fig)
    else:
        img = None
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="e.g. outputs/jepa_lineex (generator out_dir)")
    ap.add_argument("--manifest", type=str, required=True, help="path to manifest.jsonl (relative ok)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--K", type=int, default=64, help="number of x-bins / y-values to predict")
    ap.add_argument("--line_id", type=int, default=0)

    # x/y ranges (should match generator)
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=10.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=100.0)
    ap.add_argument("--normalize_y_to_0_1", action="store_true")

    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # encoder / model
    ap.add_argument("--checkpoint", type=str, default="", help="optional path to JEPA checkpoint (.pth.tar)")
    ap.add_argument("--model_name", type=str, default="vit_large")
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=1)
    ap.add_argument("--tubelet_size", type=int, default=1)
    ap.add_argument("--use_sdpa", action="store_true")
    ap.add_argument("--pool", type=str, default="mean", choices=["cls", "mean"])
    ap.add_argument("--dropout", type=float, default=0.0)

    # output
    ap.add_argument("--save_name", type=str, default="valuehead.pt")

    # wandb
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="lineex-valuehead")
    ap.add_argument("--wandb_run_name", type=str, default="valuehead_run")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_log_preview_every", type=int, default=200)

    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    out_dir = args.out_dir
    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        # allow passing "manifest.jsonl" if located inside out_dir
        # or relative to cwd
        if os.path.exists(os.path.join(out_dir, manifest_path)):
            manifest_path = os.path.join(out_dir, manifest_path)

    # Dataset
    ds = LineExValueDataset(
        out_dir=out_dir,
        manifest_path=manifest_path,
        img_size=args.img_size,
        K=args.K,
        line_id=args.line_id,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        normalize_y_to_0_1=args.normalize_y_to_0_1,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Build encoder (we only need encoder; predictor unused)
    encoder, _predictor = init_video_model(
        device=device,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        model_name=args.model_name,
        crop_size=args.img_size,
        pred_depth=1,          # unused
        pred_embed_dim=384,    # unused
        uniform_power=True,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=args.use_sdpa,
    )
    encoder.eval()  # keep frozen
    for p in encoder.parameters():
        p.requires_grad = False

    # determine embed_dim
    # MultiMaskWrapper(backbone) -> encoder.backbone has embed_dim
    embed_dim = getattr(encoder.backbone, "embed_dim", None)
    if embed_dim is None:
        raise RuntimeError("Could not determine encoder embed_dim (encoder.backbone.embed_dim missing).")

    # load weights if provided
    if args.checkpoint:
        try_load_encoder_from_checkpoint(encoder, args.checkpoint)

    # Value head
    head = ValueHead(embed_dim=embed_dim, K=args.K, pool=args.pool, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device.type == "cuda"))

    # loss: regression
    loss_fn = nn.SmoothL1Loss()

    # save path
    save_path = os.path.join(out_dir, args.save_name)

    # wandb init
    use_wandb = bool(args.use_wandb and WANDB_AVAILABLE)
    if args.use_wandb and not WANDB_AVAILABLE:
        print("[WARN] wandb requested but not installed. Continuing without wandb.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        print("[INFO] WandB enabled.")
    else:
        print("[INFO] WandB disabled.")

    # Train
    head.train()
    global_step = 0

    for ep in range(1, args.epochs + 1):
        running = 0.0
        n = 0

        for x, tgt, sid in dl:
            x = x.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                tokens = encoder(x)  # [B,N,D]

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device.type == "cuda")):
                pred = head(tokens)      # [B,K]
                loss = loss_fn(pred, tgt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())
            n += 1
            global_step += 1

            if use_wandb:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": opt.param_groups[0]["lr"],
                    "step": global_step,
                    "epoch": ep,
                })

                # optional preview plot every N steps (first sample of batch)
                if args.wandb_log_preview_every > 0 and (global_step % args.wandb_log_preview_every == 0):
                    img = make_wandb_preview(
                        pred=pred[0],
                        tgt=tgt[0],
                        step=global_step,
                        title=f"pred_vs_target/{sid[0]}",
                    )
                    if img is not None:
                        wandb.log({"preview/pred_vs_target": img, "step": global_step})

        avg = running / max(1, n)
        print(f"[epoch {ep:03d}/{args.epochs}] loss={avg:.6f}")

        # epoch log + save
        torch.save({
            "valuehead": head.state_dict(),
            "K": args.K,
            "pool": args.pool,
            "embed_dim": embed_dim,
            "epoch": ep,
            "args": vars(args),
        }, save_path)

        if use_wandb:
            wandb.log({"train/loss_epoch": avg, "epoch": ep})
            # also upload latest checkpoint as artifact (optional but handy)
            art = wandb.Artifact("valuehead_latest", type="model")
            art.add_file(save_path)
            wandb.log_artifact(art)

    print(f"[DONE] saved valuehead to: {save_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()