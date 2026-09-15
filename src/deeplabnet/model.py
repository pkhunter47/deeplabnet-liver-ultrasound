"""Readable reference implementation of the disclosed DeepLabNet design.

This is not the original experiment source. See docs/CODE_STATUS.md.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DirectionalHaar(nn.Module):
    """Extract absolute LH, HL and HH responses from a grayscale image."""

    def __init__(self) -> None:
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) / 2.0
        self.register_buffer("filters", filters)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        return F.conv2d(gray, self.filters, stride=2).abs()


class DeepLabNet(nn.Module):
    """DeepLabv3 baseline with entropy-guided residual refinement."""

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        base = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=1,
            aux_loss=False,
        )
        self.backbone = base.backbone
        self.classifier = base.classifier
        self.haar = DirectionalHaar()
        self.refinement = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.mask_residual = nn.Conv2d(16, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(16, 1, kernel_size=1)
        self.distance_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        size = image.shape[-2:]
        features = self.backbone(image)["out"]
        coarse = F.interpolate(
            self.classifier(features), size=size, mode="bilinear", align_corners=False
        )
        probability = coarse.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log())
        haar = F.interpolate(self.haar(image), size=size, mode="bilinear", align_corners=False)
        refined_features = self.refinement(torch.cat([coarse, haar, entropy], dim=1))
        refined = coarse + self.mask_residual(refined_features)
        return {
            "out": refined,
            "coarse": coarse,
            "boundary": self.boundary_head(refined_features),
            "distance": self.distance_head(refined_features).tanh(),
        }
