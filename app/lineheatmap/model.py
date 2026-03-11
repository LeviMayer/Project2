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


class MultiHeadDecoder(nn.Module):
    """
    Shared decoder trunk + two separate prediction heads:
      - line head
      - point head

    Input:  [B, D, H_patch, W_patch]
    Output: [B, 2, H, W]
        channel 0 -> line heatmap
        channel 1 -> point heatmap
    """

    def __init__(self, in_dim: int, image_size: int = 224):
        super().__init__()
        self.image_size = image_size

        # shared trunk
        self.trunk1 = ConvBlock(in_dim, 256)
        self.trunk2 = ConvBlock(256, 128)
        self.trunk3 = ConvBlock(128, 64)

        # line branch
        self.line_head = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        # point branch
        self.point_head = nn.Sequential(
            ConvBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk1(x)
        x = self.trunk2(x)
        x = self.trunk3(x)

        line = self.line_head(x)
        point = self.point_head(x)

        line = F.interpolate(
            line,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        point = F.interpolate(
            point,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        out = torch.cat([line, point], dim=1)  # [B,2,H,W]
        return out


class LineHeatmapModel(nn.Module):
    """
    Model that combines a pretrained JEPA encoder with a multi-head decoder.

    Pipeline:
        Image
          ↓
        JEPA Encoder (ViT)
          ↓
        Patch Features
          ↓
        Reshape to feature map
          ↓
        Multi-head Decoder
          ↓
        Heatmaps

    Output channels:
        channel 0 -> line_heatmap
        channel 1 -> point_heatmap
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 16,
        out_channels: int = 2,
    ):
        super().__init__()

        if out_channels != 2:
            raise ValueError(
                f"MultiHead LineHeatmapModel expects out_channels=2, got {out_channels}"
            )

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.h_patch = image_size // patch_size
        self.w_patch = image_size // patch_size

        self.decoder = MultiHeadDecoder(
            in_dim=embed_dim,
            image_size=image_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            out: [B, 2, H, W]
                 channel 0 -> line heatmap
                 channel 1 -> point heatmap
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