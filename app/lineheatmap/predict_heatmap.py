import argparse
from pathlib import Path

import torch
import yaml
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
        out_channels=2,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()
    return model


def save_visualization(
    image,
    gt_line_heatmap,
    pred_line_heatmap,
    gt_point_heatmap,
    pred_point_heatmap,
    out_path,
):
    image_np = image.cpu().permute(1, 2, 0).numpy()

    gt_line_np = gt_line_heatmap.cpu().squeeze().numpy()
    pred_line_np = pred_line_heatmap.cpu().squeeze().numpy()

    gt_point_np = gt_point_heatmap.cpu().squeeze().numpy()
    pred_point_np = pred_point_heatmap.cpu().squeeze().numpy()

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))

    # Row 1: line heatmap
    ax[0, 0].imshow(image_np)
    ax[0, 0].set_title("image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(gt_line_np, cmap="hot")
    ax[0, 1].set_title("gt_line_heatmap")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(pred_line_np, cmap="hot")
    ax[0, 2].set_title("pred_line_heatmap")
    ax[0, 2].axis("off")

    ax[0, 3].imshow(image_np)
    ax[0, 3].imshow(pred_line_np, cmap="hot", alpha=0.5)
    ax[0, 3].set_title("line overlay")
    ax[0, 3].axis("off")

    # Row 2: point heatmap
    ax[1, 0].imshow(image_np)
    ax[1, 0].set_title("image")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(gt_point_np, cmap="hot")
    ax[1, 1].set_title("gt_point_heatmap")
    ax[1, 1].axis("off")

    ax[1, 2].imshow(pred_point_np, cmap="hot")
    ax[1, 2].set_title("pred_point_heatmap")
    ax[1, 2].axis("off")

    ax[1, 3].imshow(image_np)
    ax[1, 3].imshow(pred_point_np, cmap="hot", alpha=0.5)
    ax[1, 3].set_title("point overlay")
    ax[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lineheatmap/lineheatmap_vitl16.yaml")
    parser.add_argument("--checkpoint", default="outputs/lineheatmap_vitl16/best.pth")
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--start_index", default=0, type=int)
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

    end_idx = min(args.start_index + args.num_samples, len(dataset))

    with torch.no_grad():
        for i in range(args.start_index, end_idx):
            sample = dataset[i]

            image = sample["image"].unsqueeze(0).to(device)
            gt_line_heatmap = sample["line_heatmap"]
            gt_point_heatmap = sample["point_heatmap"]
            sample_id = sample.get("id", f"sample_{i:04d}")

            logits = model(image)                  # [1,2,H,W]
            pred = torch.sigmoid(logits).cpu()    # [1,2,H,W]

            pred_line_heatmap = pred[:, 0:1]
            pred_point_heatmap = pred[:, 1:2]

            save_path = out_dir / f"{sample_id}.png"

            save_visualization(
                sample["image"],
                gt_line_heatmap,
                pred_line_heatmap,
                gt_point_heatmap,
                pred_point_heatmap,
                save_path,
            )

            print("saved:", save_path)


if __name__ == "__main__":
    main()