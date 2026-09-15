"""Train the manuscript-derived DeepLabNet reference on paired images/masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from src.deeplabnet import DeepLabNet


class PairedMaskDataset(Dataset):
    def __init__(self, root: Path, size: int = 352) -> None:
        self.image_dir, self.mask_dir, self.size = root / "images", root / "masks", size
        self.images = sorted(p for p in self.image_dir.iterdir() if p.is_file())
        masks = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}
        self.pairs = [(p, masks[p.stem]) for p in self.images if p.stem in masks]
        if not self.pairs:
            raise ValueError(f"No matching image/mask stems found under {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize((self.size, self.size))
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_array = (np.asarray(mask) > 127).astype(np.float32)
        inside = distance_transform_edt(mask_array)
        outside = distance_transform_edt(1.0 - mask_array)
        signed = inside - outside
        signed /= max(float(np.abs(signed).max()), 1.0)
        return image_tensor, torch.from_numpy(mask_array).unsqueeze(0), torch.from_numpy(signed).float().unsqueeze(0)


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for image, mask, _ in loader:
            prediction = model(image.to(device))["out"].sigmoid() > 0.5
            target = mask.to(device).bool()
            intersection = (prediction & target).sum((1, 2, 3)).float()
            scores.extend(((2 * intersection + 1) / (prediction.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).cpu().tolist())
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/deeplabnet.pt"))
    parser.add_argument("--pretrained-backbone", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PairedMaskDataset(args.data_root / "train", args.size), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(PairedMaskDataset(args.data_root / "val", args.size), batch_size=args.batch_size, num_workers=2)
    model = DeepLabNet(args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for image, mask, distance in tqdm(train_loader, desc=f"epoch {epoch}"):
            image, mask, distance = image.to(device), mask.to(device), distance.to(device)
            output = model(image)
            loss = (
                bce(output["out"], mask)
                + dice_loss(output["out"], mask)
                + 0.25 * bce(output["coarse"], mask)
                + 0.20 * bce(output["boundary"], boundary_target(mask))
                + 0.10 * F.smooth_l1_loss(output["distance"], distance)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_dice = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_dice": val_dice, "args": vars(args)}, args.output)


if __name__ == "__main__":
    main()
