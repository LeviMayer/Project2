import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """
    Simple upsampling decoder that converts ViT patch features
    into full-resolution heatmaps.

    Input:  [B, D, H_patch, W_patch]
    Output: [B, C, H, W]
    """

    def __init__(self, in_dim: int, out_channels: int = 1, image_size: int = 224, patch_size: int = 16):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(in_dim, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.head(x)

        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return x


class LineHeatmapModel(nn.Module):
    """
    Model that combines a pretrained JEPA encoder with a heatmap decoder.

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
        Heatmaps

    For the 2-head setup:
        channel 0 -> line_heatmap
        channel 1 -> point_heatmap
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 16,
        out_channels: int = 1,
    ):
        super().__init__()

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.h_patch = image_size // patch_size
        self.w_patch = image_size // patch_size

        self.decoder = SimpleDecoder(
            in_dim=embed_dim,
            out_channels=out_channels,
            image_size=image_size,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            out: [B, out_channels, H, W]

        Example:
            out_channels=1 -> [B,1,H,W]
            out_channels=2 -> [B,2,H,W]
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