"""
plot_segmentation.py — renders the White/Red/Blue overlay visualization
style used in paper Fig. 5 and Fig. 6: White = correct, Red =
over-segmentation (predicted but not GT), Blue = under-segmentation
(GT but not predicted).

Usage:
    python visualization/plot_segmentation.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
        --dataset isic2016 --num_samples 6 --out results/isic2016_qualitative.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, get_eval_transforms
from models import build_model_from_config


def overlay_error_map(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """(H,W) binary arrays -> (H,W,3) uint8 RGB overlay:
    white = TP, red = FP (over-seg), blue = FN (under-seg), black = TN."""
    H, W = pred_mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    tp = pred_mask & gt_mask
    fp = pred_mask & ~gt_mask
    fn = ~pred_mask & gt_mask
    rgb[tp] = [255, 255, 255]
    rgb[fp] = [255, 0, 0]
    rgb[fn] = [0, 0, 255]
    return rgb


@torch.no_grad()
def make_figure(model, dataset, device, num_samples: int, out_path: str):
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    indices = np.linspace(0, len(dataset) - 1, num_samples).astype(int)
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["mask"].numpy()[0].astype(bool)

        out = model.predict(image)
        pred = (out["refined_mask"][0, 0].cpu().numpy() > 0.5)

        img_np = sample["image"].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Image" if row == 0 else "")
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("GT" if row == 0 else "")
        axes[row, 2].imshow(overlay_error_map(pred, gt))
        axes[row, 2].set_title("Ours (White=TP, Red=FP, Blue=FN)" if row == 0 else "")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved qualitative figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2016")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--out", default="results/qualitative.png")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg
    device = base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    ds_cfg = cfg["data"]["datasets"][args.dataset]
    eval_tf = get_eval_transforms(image_size=cfg["data"]["image_size"])
    dataset = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)

    model = build_model_from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    make_figure(model, dataset, device, args.num_samples, args.out)


if __name__ == "__main__":
    main()
