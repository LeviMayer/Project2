import argparse
import os
from pathlib import Path

import torch
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from app.lineheatmap.dataset import LineHeatmapDataset
from app.lineheatmap.model import LineHeatmapModel
from app.vjepa.utils import init_video_model


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg, device, checkpoint):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    encoder, _ = init_video_model(
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

    embed_dim_map = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_huge": 1280,
    }

    embed_dim = train_cfg.get(
        "embed_dim",
        embed_dim_map.get(model_cfg["model_name"], 1024),
    )

    model = LineHeatmapModel(
        encoder=encoder,
        embed_dim=embed_dim,
        image_size=data_cfg["crop_size"],
        patch_size=data_cfg["patch_size"],
        out_channels=1,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model = model.to(device)
    model.eval()

    return model


def save_visualization(image, gt_heatmap, pred_heatmap, out_path):
    image = image.cpu().permute(1, 2, 0).numpy()
    gt = gt_heatmap.cpu().squeeze().numpy()
    pred = pred_heatmap.cpu().squeeze().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))

    ax[0].imshow(image)
    ax[0].set_title("image")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="hot")
    ax[1].set_title("gt_heatmap")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="hot")
    ax[2].set_title("pred_heatmap")
    ax[2].axis("off")

    ax[3].imshow(image)
    ax[3].imshow(pred, cmap="hot", alpha=0.5)
    ax[3].set_title("overlay")
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lineheatmap/lineheatmap_vitl16.yaml")
    parser.add_argument("--checkpoint", default="outputs/lineheatmap_vitl16/best1.pth")
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--out_dir", default="outputs/predictions")

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LineHeatmapDataset(
        root_dir=cfg["data"]["root_dir"],
        manifest_path=cfg["data"]["val_manifest"],
        image_key=cfg["data"].get("image_key", "full"),
        image_size=cfg["data"]["crop_size"],
        heatmap_size=cfg["data"].get("heatmap_size", cfg["data"]["crop_size"]),
        sigma=cfg["data"].get("sigma", 2.5),
        x_min=cfg["data"].get("x_min", 0.0),
        x_max=cfg["data"].get("x_max", 10.0),
        y_min=cfg["data"].get("y_min", 0.0),
        y_max=cfg["data"].get("y_max", 100.0),
    )

    model = build_model(cfg, device, args.checkpoint)

    print("dataset size:", len(dataset))

    with torch.no_grad():
        for i in range(min(args.num_samples, len(dataset))):

            sample = dataset[i]

            image = sample["image"].unsqueeze(0).to(device)
            gt_heatmap = sample["heatmap"]

            logits = model(image)
            pred_heatmap = torch.sigmoid(logits).cpu()

            save_path = out_dir / f"sample_{i:04d}.png"

            save_visualization(
                sample["image"],
                gt_heatmap,
                pred_heatmap,
                save_path,
            )

            print("saved:", save_path)


if __name__ == "__main__":
    main()