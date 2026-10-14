import pytest

from visionguard.device import describe_device, select_device


def test_cpu_device_is_always_available():
    assert select_device("cpu") == "cpu"
    assert describe_device("cpu") == "CPU"


def test_auto_device_matches_torch_cuda_availability():
    import torch

    assert select_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_invalid_device_preference_is_rejected():
    with pytest.raises(ValueError, match="Device preference"):
        select_device("tpu")
