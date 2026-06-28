from __future__ import annotations


def get_device(preferred: str | None = None):
    """Return a torch.device, falling back to CPU when CUDA is unavailable."""
    import torch

    requested = preferred or "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def print_cuda_memory(prefix: str = "") -> None:
    """Print current CUDA memory usage when CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        print(f"{prefix}CUDA unavailable")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{prefix}CUDA memory allocated={allocated:.1f}MB reserved={reserved:.1f}MB")

