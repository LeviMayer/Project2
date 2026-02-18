import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.lineex_dataset import LineExDataset
from app.vjepa.utils import init_video_model

class ValueHead(nn.Module):
    def __init__(self, in_dim, T=64, Lmax=3, hidden=1024):
        super().__init__()
        self.T, self.Lmax = T, Lmax
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, Lmax * T),
        )

    def forward(self, feats):  # [B,D]
        y = self.net(feats)    # [B,Lmax*T]
        return y.view(-1, self.Lmax, self.T)

class LineExModel(nn.Module):
    def __init__(self, encoder_wrapper, T=64, Lmax=3):
        super().__init__()
        self.backbone = encoder_wrapper.backbone
        self.head = ValueHead(self.backbone.embed_dim, T=T, Lmax=Lmax)

    def forward(self, x):      # [B,3,224,224]
        tok = self.backbone(x) # [B,N,D]
        feats = tok.mean(dim=1)
        return self.head(feats)

def load_encoder_only(ckpt_path, encoder_wrapper):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder_wrapper.load_state_dict(ckpt["encoder"], strict=True)
    print("Loaded encoder from", ckpt_path)

def masked_mse(pred, target, line_mask):
    # pred/target [B,L,T], line_mask [B,L]
    m = line_mask.unsqueeze(-1)
    diff2 = (pred - target) ** 2
    diff2 = diff2 * m
    denom = m.sum() * pred.size(-1) + 1e-8
    return diff2.sum() / denom

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- HIER anpassen ----------
    DATA_ROOT = "out_lineex"  # <-- dein Generator-Output mit manifest.jsonl
    CKPT = "outputs/jepa_lineex/lineex_vjepa-latest.pth.tar"  # <-- dein JEPA checkpoint
    # ----------------------------------

    T = 64
    Lmax = 3  # erst 1? dann Lmax=1; später easy auf 3/5 erhöhen

    ds = LineExDataset(root=DATA_ROOT, T=T, Lmax=Lmax, use_masked=False)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    encoder, _predictor = init_video_model(
        device=device,
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        model_name="vit_large",
        crop_size=224,
        pred_depth=12,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        use_sdpa=True,
    )
    load_encoder_only(CKPT, encoder)

    model = LineExModel(encoder, T=T, Lmax=Lmax).to(device)

    # Phase 1: Backbone einfrieren, nur head trainieren
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for step, batch in enumerate(dl):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        lm = batch["line_mask"].to(device, non_blocking=True)

        pred = model(x)
        loss = masked_mse(pred, y, lm)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    torch.save({"model": model.state_dict(), "T": T, "Lmax": Lmax}, "valuehead_headonly.pth")
    print("saved valuehead_headonly.pth")

if __name__ == "__main__":
    main()
