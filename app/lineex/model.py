import torch
import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self, in_dim: int, T: int = 64, Lmax: int = 1, hidden: int = 1024):
        super().__init__()
        self.T = T
        self.Lmax = Lmax
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, Lmax * T),
        )

    def forward(self, feats):  # feats: [B, D]
        out = self.mlp(feats)              # [B, Lmax*T]
        return out.view(-1, self.Lmax, self.T)

class EncoderWithHead(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, T=64, Lmax=1):
        super().__init__()
        self.encoder = encoder
        self.head = ValueHead(embed_dim, T=T, Lmax=Lmax)

    def forward(self, x):  # x: [B,3,H,W]
        tok = self.encoder(x)              # meist [B, N, D]
        if tok.dim() == 3:
            feats = tok.mean(dim=1)        # mean pool über Tokens -> [B, D]
        else:
            feats = tok                     # falls encoder schon [B,D] liefert
        y = self.head(feats)               # [B, Lmax, T]
        return y
