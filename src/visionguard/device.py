"""Device selection helpers for CPU and CUDA inference."""

from __future__ import annotations


def select_device(preference: str = "auto") -> str:
    """Select CPU or CUDA, raising a useful error for unavailable forced CUDA."""
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device preference must be 'auto', 'cpu', or 'cuda'.")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this environment. Install project requirements first."
        ) from exc
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(preference: str = "auto") -> str:
    """Return a concise description of the selected compute device."""
    device = select_device(preference)
    if device == "cpu":
        return "CPU"
    import torch
    return f"CUDA: {torch.cuda.get_device_name(0)}"
