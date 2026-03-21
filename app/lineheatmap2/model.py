import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SlotHeadDecoder(nn.Module):
    """
    Shared decoder trunk + separate heads:
      - line head:  1 channel
      - point head: K slot channels

    Input:
        [B, D, H_patch, W_patch]

    Output:
        dict with
          - line_logits:  [B, 1, H, W]
          - point_logits: [B, K, H, W]
    """

    def __init__(
        self,
        in_dim: int,
        image_size: int = 224,
        point_slots: int = 16,
    ):
        super().__init__()
        self.image_size = image_size
        self.point_slots = point_slots

        # shared trunk
        self.trunk1 = ConvBlock(in_dim, 256)
        self.trunk2 = ConvBlock(256, 128)
        self.trunk3 = ConvBlock(128, 64)

        # line branch
        self.line_head = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # point branch: slightly stronger / more separate than before
        self.point_head = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, point_slots, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.trunk1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)

        line = self.line_head(x)              # [B,1,h,w]
        point_slots = self.point_head(x)      # [B,K,h,w]

        line = F.interpolate(
            line,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        point_slots = F.interpolate(
            point_slots,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "line_logits": line,
            "point_logits": point_slots,
        }


class LineHeatmapModel(nn.Module):
    """
    Model that combines a pretrained JEPA encoder with a slot-based decoder.

    Pipeline:
        Image
          ↓
        JEPA Encoder (ViT)
          ↓
        Patch Features
          ↓
        Reshape to feature map
          ↓
        Decoder
          ↓
        line_logits  [B,1,H,W]
        point_logits [B,K,H,W]
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 16,
        point_slots: int = 16,
    ):
        super().__init__()

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.point_slots = point_slots

        self.h_patch = image_size // patch_size
        self.w_patch = image_size // patch_size

        self.decoder = SlotHeadDecoder(
            in_dim=embed_dim,
            image_size=image_size,
            point_slots=point_slots,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            {
                "line_logits":  [B,1,H,W],
                "point_logits": [B,K,H,W],
            }
        """
        B = x.shape[0]

        feats = self.encoder(x)

        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.dim() == 3:
            # [B, N, D] -> [B, D, H_patch, W_patch]
            feats = feats.transpose(1, 2)
            feats = feats.reshape(B, self.embed_dim, self.h_patch, self.w_patch)
        elif feats.dim() != 4:
            raise ValueError(f"Unexpected encoder output shape: {feats.shape}")

        out = self.decoder(feats)
        return out