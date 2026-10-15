"""Frozen pretrained ResNet18 features for PatchCore-inspired patch scoring."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from visionguard.device import select_device

RESNET18_PRETRAINED_WEIGHTS = "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"


class ModelSetupError(RuntimeError):
    """Raised when the required pretrained feature extractor cannot be loaded."""


def _load_pretrained_backbone() -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except (ImportError, RuntimeError) as exc:
        raise ModelSetupError(
            "TorchVision ResNet18 is unavailable. Install the project's matched "
            "PyTorch and TorchVision versions."
        ) from exc

    try:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except (OSError, RuntimeError) as exc:
        raise ModelSetupError(
            "Pretrained ResNet18 weights could not be loaded. Check the local "
            "TorchVision weight cache or allow the official weights to download. "
            "Random weights are not used as a fallback."
        ) from exc


class ResNet18PatchFeatureExtractor(nn.Module):
    """Extract normalized layer2/layer3 patch embeddings from RGB image batches.

    Inputs are float32 tensors shaped ``[batch, 3, height, width]`` with values
    in ``[0, 1]``. Outputs are ``[batch, patch_height, patch_width, channels]``.
    The default path always loads pretrained ImageNet weights. ``backbone`` is
    an injection point for small deterministic test networks.
    """

    def __init__(
        self,
        *,
        device_preference: str = "auto",
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(select_device(device_preference))
        self.backbone = backbone if backbone is not None else _load_pretrained_backbone()
        required_layers = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
        missing_layers = [name for name in required_layers if not hasattr(self.backbone, name)]
        if missing_layers:
            raise ModelSetupError(
                "The feature backbone is missing required ResNet stages: "
                + ", ".join(missing_layers)
            )

        self.backbone.to(self.device)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(self.device),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(self.device),
            persistent=False,
        )
        super().train(False)

    def train(self, mode: bool = True) -> ResNet18PatchFeatureExtractor:
        """Keep the frozen backbone and batch-normalization layers in eval mode."""
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1] != 3
            or images.shape[2] == 0
            or images.shape[3] == 0
        ):
            raise ValueError("images must be a non-empty [batch, 3, height, width] tensor.")
        if images.dtype != torch.float32:
            raise ValueError("images must use float32 values scaled to [0, 1].")
        if not torch.isfinite(images).all().item():
            raise ValueError("images must contain only finite values.")
        if images.numel() and (images.min().item() < 0 or images.max().item() > 1):
            raise ValueError("images must be scaled to the [0, 1] range.")

        inputs = images.to(self.device)
        inputs = (inputs - self.image_mean) / self.image_std
        features = self.backbone
        outputs = features.conv1(inputs)
        outputs = features.bn1(outputs)
        outputs = features.relu(outputs)
        outputs = features.maxpool(outputs)
        outputs = features.layer1(outputs)
        layer2 = features.layer2(outputs)
        layer3 = features.layer3(layer2)

        layer2_patches = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        layer3_patches = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        layer3_patches = F.interpolate(
            layer3_patches,
            size=layer2_patches.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        embeddings = torch.cat((layer2_patches, layer3_patches), dim=1)
        return embeddings.permute(0, 2, 3, 1).contiguous()
