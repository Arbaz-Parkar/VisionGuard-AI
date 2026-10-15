import numpy as np
import pytest
import torch
from torch import nn

from visionguard.model import ModelSetupError, ResNet18PatchFeatureExtractor
from visionguard.model import features as feature_module


class TinyBackbone(nn.Module):
    """Small deterministic-shaped network with the ResNet stages used by the extractor."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1)


def test_extractor_combines_local_layer2_and_layer3_patch_features_on_cpu():
    extractor = ResNet18PatchFeatureExtractor(
        device_preference="cpu", backbone=TinyBackbone()
    )
    images = torch.rand((2, 3, 64, 64), dtype=torch.float32)

    embeddings = extractor(images)

    assert extractor.device.type == "cpu"
    assert embeddings.shape == (2, 8, 8, 14)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()
    assert not embeddings.requires_grad
    assert not extractor.backbone.training
    assert all(not parameter.requires_grad for parameter in extractor.backbone.parameters())


def test_extractor_stays_in_eval_mode_and_accepts_numpy_based_input():
    extractor = ResNet18PatchFeatureExtractor(
        device_preference="cpu", backbone=TinyBackbone()
    )
    extractor.train()
    images = torch.from_numpy(np.zeros((1, 3, 64, 64), dtype=np.float32))

    embeddings = extractor(images)

    assert not extractor.training
    assert not extractor.backbone.training
    assert embeddings.shape[0] == 1


def test_default_backbone_reports_missing_weights_without_random_fallback(monkeypatch):
    import torchvision.models

    def missing_weights(**kwargs):
        raise OSError("synthetic missing weights")

    monkeypatch.setattr(torchvision.models, "resnet18", missing_weights)

    with pytest.raises(ModelSetupError, match="Random weights are not used"):
        feature_module._load_pretrained_backbone()


def test_default_backbone_uses_a_pinned_torchvision_weight_version(monkeypatch):
    import torchvision.models

    marker = nn.Identity()
    received = {}

    def load_backbone(**kwargs):
        received.update(kwargs)
        return marker

    monkeypatch.setattr(torchvision.models, "resnet18", load_backbone)

    backbone = feature_module._load_pretrained_backbone()

    assert backbone is marker
    assert received["weights"] is torchvision.models.ResNet18_Weights.IMAGENET1K_V1


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((3, 64, 64), dtype=torch.float32),
        torch.zeros((1, 1, 64, 64), dtype=torch.float32),
        torch.zeros((1, 3, 64, 64), dtype=torch.uint8),
        torch.full((1, 3, 64, 64), 2.0, dtype=torch.float32),
        torch.full((1, 3, 64, 64), float("nan"), dtype=torch.float32),
    ],
)
def test_extractor_rejects_invalid_image_batches(images):
    extractor = ResNet18PatchFeatureExtractor(
        device_preference="cpu", backbone=TinyBackbone()
    )

    with pytest.raises(ValueError):
        extractor(images)
