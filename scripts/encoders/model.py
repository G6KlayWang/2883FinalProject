"""Random ResNet-18 and unnormalized 128-D projector, shared by every arm."""

import torch
from torch import nn
from torchvision.models import resnet18
from scripts.encoders.rng import keyed_seed


class VisualEncoder(nn.Module):
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 96, 96):
            raise ValueError("Encoder requires (B,3,96,96) images")
        return self.projector(self.backbone(images))


def build_encoder(seed: int, hidden: int = 2048) -> VisualEncoder:
    # Initialization never advances augmentation, target, direction or global RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(keyed_seed(seed, "initialization"))
        return VisualEncoder(hidden)


def forward_views(
    model: nn.Module, view1: torch.Tensor, view2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate forwards, identical BatchNorm convention in A/B/C/A-prime."""
    return model(view1), model(view2)
