import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import wandb

from lineex_dataset import LineExDataset
from app.vjepa.utils import init_video_model


# ----------------------------
# Config
# ----------------------------
@dataclass
class Cfg:
    data_root: str = "out_lineex"  # generator output with manifest.jsonl
    ckpt: str = "outputs/jepa_lineex/lineex_vjepa-latest.pth.tar"

    out_path: str = "valuehead_headonly.pth"

    img_size: int = 224
    model_name: str = "vit_large"
    patch_size: int = 16
    num_frames: int = 1
    tubelet_size: int = 1

    T: int = 64
    Lmax: int = 1  # start with 1, later set 3/5

    batch_size: int = 32
    num_workers: int = 4

    lr: float = 1e-3
    weight_decay: float = 1e-4
    steps: int = 3000
    log_every: int = 50
    val_every: int = 200

    seed: int = 42
    wandb_project: str = "lineex-valuehead"
    wandb_run_name: str = "headonly"


# ----------------------------
# Model
# ----------------------------
class ValueHead(nn.Module):
    def __init__(self, in_dim, T=64, Lmax=3, hidden=1024):
        super().__init__()
        self.T, self.Lmax = T, Lmax
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, Lmax * T),
        )

    def forward(self, feats):  # [B, D]
        y = self.net(feats)    # [B, Lmax*T]
        return y.view(-1, self.Lmax, self.T)


class LineExModel(nn.Module):
    def __init__(self, encoder_wrapper, T=64, Lmax=3):
        super().__init__()
        # encoder_wrapper is MultiMaskWrapper, we only need its backbone (ViT)
        self.backbone = encoder_wrapper.backbone
        self.head = ValueHead(self.backbone.embed_dim, T=T, Lmax=Lmax)

    def forward(self, x):      # [B, 3, 224, 224]
        tok = self.backbone(x) # [B, N, D]
        feats = tok.mean(dim=1)
        return self.head(feats)


def load_encoder_only(ckpt_path, encoder_wrapper):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder_wrapper.load_state_dict(ckpt["encoder"], strict=True)
    print("Loaded encoder from", ckpt_path)


def masked_mse(pred, target, line_mask):
    """
    pred/target: [B, L, T]
    line_mask:   [B, L] (1 for valid line, 0 for not present)
    """
    m = line_mask.unsqueeze(-1)                 # [B, L, 1]
    diff2 = (pred - target) ** 2
    diff2 = diff2 * m
    denom = m.sum() * pred.size(-1) + 1e-8
    return diff2.sum() / denom


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    losses = []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        lm = batch["line_mask"].to(device, non_blocking=True)
        pred = model(x)
        loss = masked_mse(pred, y, lm)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    cfg = Cfg()

    # Repro
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B (set WANDB_MODE=offline on HPC if no internet)
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=vars(cfg),
    )

    # Dataset + split
    ds = LineExDataset(root=cfg.data_root, T=cfg.T, Lmax=cfg.Lmax, use_masked=False)
    n_val = max(1, int(0.05 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    # Backbone init
    encoder, _ = init_video_model(
        device=device,
        patch_size=cfg.patch_size,
        num_frames=cfg.num_frames,
        tubelet_size=cfg.tubelet_size,
        model_name=cfg.model_name,
        crop_size=cfg.img_size,
        pred_depth=12,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        use_sdpa=True,
    )
    load_encoder_only(cfg.ckpt, encoder)

    model = LineExModel(encoder, T=cfg.T, Lmax=cfg.Lmax).to(device)

    # Phase 1: freeze backbone, train head only
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train loop (step-based)
    step = 0
    train_it = iter(train_dl)

    while step < cfg.steps:
        try:
            batch = next(train_it)
        except StopIteration:
            train_it = iter(train_dl)
            batch = next(train_it)

        x = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        lm = batch["line_mask"].to(device, non_blocking=True)

        pred = model(x)
        loss = masked_mse(pred, y, lm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0:
            wandb.log({"train/loss": loss.item()}, step=step)
            print(f"step {step} train_loss {loss.item():.4f}")

        if step % cfg.val_every == 0 and step > 0:
            val_loss = evaluate(model, val_dl, device)
            wandb.log({"val/loss": val_loss}, step=step)
            print(f"step {step} val_loss {val_loss:.4f}")

        step += 1

    # Save
    save = {
        "model": model.state_dict(),
        "T": cfg.T,
        "Lmax": cfg.Lmax,
        "ckpt_encoder": cfg.ckpt,
        "cfg": vars(cfg),
    }
    torch.save(save, cfg.out_path)
    print("saved", cfg.out_path)

    wandb.finish()


if __name__ == "__main__":
    main()