import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Assumes you will add these files next:
#   app/lineheatmap/dataset.py
#   app/lineheatmap/model.py
from app.lineheatmap.dataset import LineHeatmapDataset
from app.lineheatmap.model import LineHeatmapModel
from app.vjepa.utils import init_video_model


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: [B, 1, H, W]
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.eps) / (denom + self.eps))
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val: float,
    config: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(ckpt, save_path)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        images, heatmaps = batch["image"].to(device), batch["heatmap"].to(device)
        logits = model(images)
        loss = criterion(logits, heatmaps)
        losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def maybe_load_encoder_checkpoint(encoder: nn.Module, ckpt_path: str) -> None:
    """
    Loads a JEPA checkpoint as best effort.
    Tries common key layouts without assuming a single exact format.
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[WARN] No encoder checkpoint loaded. ckpt_path={ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = None

    # Common possibilities
    for key in ["encoder", "target_encoder", "model", "state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break

    if state is None and isinstance(ckpt, dict):
        state = ckpt

    # Strip possible prefixes
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ["module.", "backbone.", "encoder."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    print(f"[INFO] Loaded encoder checkpoint: {ckpt_path}")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing:
        print("[INFO] First missing keys:", missing[:10])
    if unexpected:
        print("[INFO] First unexpected keys:", unexpected[:10])


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    encoder, _predictor = init_video_model(
        uniform_power=model_cfg.get("uniform_power", True),
        use_mask_tokens=model_cfg.get("use_mask_tokens", True),
        num_mask_tokens=1,
        zero_init_mask_tokens=model_cfg.get("zero_init_mask_tokens", True),
        device=device,
        patch_size=data_cfg["patch_size"],
        num_frames=data_cfg.get("num_frames", 1),
        tubelet_size=data_cfg.get("tubelet_size", 1),
        model_name=model_cfg["model_name"],
        crop_size=data_cfg["crop_size"],
        pred_depth=model_cfg.get("pred_depth", 12),
        pred_embed_dim=model_cfg.get("pred_embed_dim", 384),
        use_sdpa=cfg.get("meta", {}).get("use_sdpa", True),
    )

    ckpt_path = train_cfg.get("encoder_checkpoint", None)
    maybe_load_encoder_checkpoint(encoder, ckpt_path)

    if train_cfg.get("freeze_encoder", True):
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    embed_dim_map = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_huge": 1280,
    }
    embed_dim = train_cfg.get("embed_dim", embed_dim_map.get(model_cfg["model_name"], 1024))

    model = LineHeatmapModel(
        encoder=encoder,
        embed_dim=embed_dim,
        image_size=data_cfg["crop_size"],
        patch_size=data_cfg["patch_size"],
        out_channels=1,
    )
    return model.to(device)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    train_ds = LineHeatmapDataset(
        root_dir=data_cfg["root_dir"],
        manifest_path=data_cfg["train_manifest"],
        image_key=data_cfg.get("image_key", "full"),
        image_size=data_cfg["crop_size"],
        heatmap_size=data_cfg.get("heatmap_size", data_cfg["crop_size"]),
        sigma=data_cfg.get("sigma", 2.5),
        x_min=data_cfg.get("x_min", 0.0),
        x_max=data_cfg.get("x_max", 10.0),
        y_min=data_cfg.get("y_min", 0.0),
        y_max=data_cfg.get("y_max", 100.0),
    )

    val_ds = LineHeatmapDataset(
        root_dir=data_cfg["root_dir"],
        manifest_path=data_cfg["val_manifest"],
        image_key=data_cfg.get("image_key", "full"),
        image_size=data_cfg["crop_size"],
        heatmap_size=data_cfg.get("heatmap_size", data_cfg["crop_size"]),
        sigma=data_cfg.get("sigma", 2.5),
        x_min=data_cfg.get("x_min", 0.0),
        x_max=data_cfg.get("x_max", 10.0),
        y_min=data_cfg.get("y_min", 0.0),
        y_max=data_cfg.get("y_max", 100.0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_mem", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=max(1, data_cfg.get("num_workers", 4) // 2),
        pin_memory=data_cfg.get("pin_mem", True),
        drop_last=False,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    log_every: int,
) -> Tuple[float, int]:
    model.train()
    running = 0.0
    steps = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["heatmap"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, heatmaps)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        steps += 1

        if step % log_every == 0:
            print(
                f"[epoch {epoch:03d} | step {step:05d}/{len(loader):05d}] "
                f"loss={running / steps:.5f}"
            )

    return running / max(steps, 1), steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["logging"]["folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg, device)

    # Build optimizer only over trainable params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    criterion = DiceBCELoss(
        bce_weight=cfg["train"].get("bce_weight", 1.0),
        dice_weight=cfg["train"].get("dice_weight", 1.0),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    global_step = 0
    epochs = cfg["train"].get("epochs", 20)
    log_every = cfg["train"].get("log_every", 25)
    eval_every = cfg["train"].get("eval_every", 1)

    with open(out_dir / "train_config_dump.json", "w") as f:
        json.dump(cfg, f, indent=2)

    for epoch in range(1, epochs + 1):
        train_loss, n_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_every=log_every,
        )
        global_step += n_steps

        print(f"[INFO] epoch={epoch} train_loss={train_loss:.6f}")

        if epoch % eval_every == 0:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"[INFO] epoch={epoch} val_loss={val_loss:.6f}")

            save_checkpoint(
                save_path=str(out_dir / "latest.pth"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_val=best_val,
                config=cfg,
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    save_path=str(out_dir / "best.pth"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_val=best_val,
                    config=cfg,
                )
                print(f"[INFO] new best checkpoint saved (val={best_val:.6f})")

    print("[INFO] training finished")


if __name__ == "__main__":
    main()
