import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb

from app.lineheatmap2.dataset import LineHeatmapDataset
from app.lineheatmap2.model import LineHeatmapModel
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


class MaskedPointSlotBCEMSELoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        mse_weight: float = 1.0,
        invalid_weight: float = 0.25,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_weight = float(bce_weight)
        self.mse_weight = float(mse_weight)
        self.invalid_weight = float(invalid_weight)
        self.eps = eps

    def forward(self, logits, targets, valid_mask):
        valid_mask = valid_mask.float()  # [B,K]
        invalid_mask = 1.0 - valid_mask

        bce = self.bce(logits, targets).mean(dim=(2, 3))  # [B,K]
        probs = torch.sigmoid(logits)
        mse = ((probs - targets) ** 2).mean(dim=(2, 3))   # [B,K]

        valid_loss = (self.bce_weight * bce + self.mse_weight * mse) * valid_mask

        # invalid slots should stay empty
        invalid_target = torch.zeros_like(targets)
        invalid_bce = self.bce(logits, invalid_target).mean(dim=(2, 3))
        invalid_mse = ((probs - invalid_target) ** 2).mean(dim=(2, 3))

        invalid_loss = (self.bce_weight * invalid_bce + self.mse_weight * invalid_mse) * invalid_mask

        valid_denom = valid_mask.sum().clamp_min(self.eps)
        invalid_denom = invalid_mask.sum().clamp_min(self.eps)

        return valid_loss.sum() / valid_denom + self.invalid_weight * (invalid_loss.sum() / invalid_denom)


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
    line_criterion: nn.Module,
    point_criterion: nn.Module,
    device: torch.device,
    point_loss_weight: float = 5.0,
) -> float:
    model.eval()
    losses = []

    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        line_heatmap = batch["line_heatmap"].to(device, non_blocking=True)         # [B,1,H,W]
        point_heatmaps = batch["point_heatmaps"].to(device, non_blocking=True)     # [B,K,H,W]
        point_valid_mask = batch["point_valid_mask"].to(device, non_blocking=True) # [B,K]

        out = model(images)
        line_logits = out["line_logits"]      # [B,1,H,W]
        point_logits = out["point_logits"]    # [B,K,H,W]

        loss_line = line_criterion(line_logits, line_heatmap)
        loss_point = point_criterion(point_logits, point_heatmaps, point_valid_mask)
        loss = loss_line + point_loss_weight * loss_point

        losses.append(loss.item())

        if i % 100 == 0:
            print(f"[VAL] step {i}/{len(loader)}")

    return float(sum(losses) / max(len(losses), 1))


def maybe_load_encoder_checkpoint(encoder: nn.Module, ckpt_path: str) -> None:
    """
    Load JEPA encoder weights from checkpoint.

    init_video_model() returns a wrapper (e.g. MultiMaskWrapper),
    while the checkpoint usually stores weights for the inner ViT backbone.
    So we load into encoder.backbone if it exists.
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[WARN] No encoder checkpoint loaded. ckpt_path={ckpt_path}")
        return

    print(f"[INFO] Loading encoder checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        state_dict = ckpt["encoder"]
    elif "target_encoder" in ckpt and isinstance(ckpt["target_encoder"], dict):
        state_dict = ckpt["target_encoder"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        if nk.startswith("target_encoder."):
            nk = nk[len("target_encoder."):]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        cleaned[nk] = v

    target = encoder.backbone if hasattr(encoder, "backbone") else encoder

    missing, unexpected = target.load_state_dict(cleaned, strict=False)

    print(f"[INFO] Encoder checkpoint loaded.")
    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")
    if missing:
        print("[INFO] Example missing keys:", missing[:10])
    if unexpected:
        print("[INFO] Example unexpected keys:", unexpected[:10])


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
        point_slots=model_cfg.get("point_slots", data_cfg.get("max_points", 16)),
    )
    return model.to(device)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]

    common_kwargs = dict(
        root_dir=data_cfg["root_dir"],
        image_key=data_cfg.get("image_key", "full"),
        image_size=data_cfg["crop_size"],
        heatmap_size=data_cfg.get("heatmap_size", data_cfg["crop_size"]),
        sigma=data_cfg.get("sigma", 1.5),
        point_sigma=data_cfg.get("point_sigma", 0.6),
        max_points=data_cfg.get("max_points", 16),
        x_min=data_cfg.get("x_min", 0.0),
        x_max=data_cfg.get("x_max", 10.0),
        y_min=data_cfg.get("y_min", 0.0),
        y_max=data_cfg.get("y_max", 100.0),
    )

    train_ds = LineHeatmapDataset(
        manifest_path=data_cfg["train_manifest"],
        **common_kwargs,
    )

    val_ds = LineHeatmapDataset(
        manifest_path=data_cfg["val_manifest"],
        **common_kwargs,
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
    line_criterion: nn.Module,
    point_criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    log_every: int,
    global_step_start: int,
    point_loss_weight: float = 5.0,
) -> Tuple[float, int]:
    model.train()
    if hasattr(model, "encoder"):
        model.encoder.eval()

    running = 0.0
    steps = 0

    for step, batch in enumerate(loader, start=1):
        global_step = global_step_start + step

        images = batch["image"].to(device, non_blocking=True)
        line_heatmap = batch["line_heatmap"].to(device, non_blocking=True)         # [B,1,H,W]
        point_heatmaps = batch["point_heatmaps"].to(device, non_blocking=True)     # [B,K,H,W]
        point_valid_mask = batch["point_valid_mask"].to(device, non_blocking=True) # [B,K]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(images)

            line_logits = out["line_logits"]      # [B,1,H,W]
            point_logits = out["point_logits"]    # [B,K,H,W]

            if step == 1 and epoch == 1:
                print("DEBUG images:", images.shape)
                print("DEBUG line_heatmap:", line_heatmap.shape)
                print("DEBUG point_heatmaps:", point_heatmaps.shape)
                print("DEBUG point_valid_mask:", point_valid_mask.shape)
                print("DEBUG line_logits:", line_logits.shape)
                print("DEBUG point_logits:", point_logits.shape)

            loss_line = line_criterion(line_logits, line_heatmap)
            loss_point = point_criterion(point_logits, point_heatmaps, point_valid_mask)
            loss = loss_line + point_loss_weight * loss_point

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        steps += 1

        if step % log_every == 0:
            avg_loss = running / steps
            print(
                f"[epoch {epoch:03d} | step {step:05d}/{len(loader):05d}] "
                f"loss={avg_loss:.5f} "
                f"(line={loss_line.item():.5f}, point={loss_point.item():.5f})"
            )

            if wandb.run is not None:
                wandb.log({
                    "global_step": global_step,
                    "epoch": epoch,
                    "train/loss_step": avg_loss,
                    "train/loss_line_step": loss_line.item(),
                    "train/loss_point_step": loss_point.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                })

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

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"].get("lr", 1e-4),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    line_criterion = DiceBCELoss(
        bce_weight=cfg["train"].get("bce_weight", 1.0),
        dice_weight=cfg["train"].get("dice_weight", 1.0),
    )

    point_criterion = MaskedPointSlotBCEMSELoss(
        bce_weight=cfg["train"].get("point_bce_weight", 1.0),
        mse_weight=cfg["train"].get("point_mse_weight", 1.0),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    global_step = 0
    epochs = cfg["train"].get("epochs", 20)
    log_every = cfg["train"].get("log_every", 25)
    eval_every = cfg["train"].get("eval_every", 1)
    point_loss_weight = cfg["train"].get("point_loss_weight", 5.0)

    with open(out_dir / "train_config_dump.json", "w") as f:
        json.dump(cfg, f, indent=2)

    wandb.init(
        project=cfg["logging"].get("wandb_project", "linechart-extraction"),
        name=cfg["logging"].get("run_name", "lineheatmap_run"),
        config=cfg,
    )

    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="epoch")

    for epoch in range(1, epochs + 1):
        train_loss, n_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            line_criterion=line_criterion,
            point_criterion=point_criterion,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_every=log_every,
            global_step_start=global_step,
            point_loss_weight=point_loss_weight,
        )
        global_step += n_steps

        print(f"[INFO] epoch={epoch} train_loss={train_loss:.6f}")

        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss_epoch": train_loss,
            })

        if epoch % eval_every == 0:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                line_criterion=line_criterion,
                point_criterion=point_criterion,
                device=device,
                point_loss_weight=point_loss_weight,
            )
            print(f"[INFO] epoch={epoch} val_loss={val_loss:.6f}")

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "val/loss": val_loss,
                })

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

                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "val/best_loss": best_val,
                    })

    print("[INFO] training finished")
    wandb.finish()


if __name__ == "__main__":
    main()