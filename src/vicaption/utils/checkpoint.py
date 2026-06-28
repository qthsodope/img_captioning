from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str,
    connector,
    optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Save Connector-only weights and optimizer state."""
    import torch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "connector": connector.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        checkpoint_path,
    )


def save_lora_connector_checkpoint(
    path: str | Path,
    connector,
    decoder,
    optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Save Connector plus decoder LoRA adapter weights."""
    import torch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "connector": connector.state_dict(),
            "decoder_lora": decoder.lora_state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        checkpoint_path,
    )


def load_connector_checkpoint(path: str, connector, map_location: str = "cpu") -> dict[str, Any]:
    """Load Connector weights from a checkpoint and return the checkpoint payload."""
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("connector", checkpoint)
    connector.load_state_dict(state_dict)
    return checkpoint


def load_lora_connector_checkpoint(
    path: str | Path,
    connector,
    decoder,
    optimizer=None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load Connector plus decoder LoRA adapter weights."""
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    connector.load_state_dict(checkpoint["connector"])
    decoder_lora = checkpoint.get("decoder_lora")
    if decoder_lora is None:
        raise ValueError(f"Checkpoint does not contain decoder_lora weights: {path}")
    decoder.load_lora_state_dict(decoder_lora)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def load_training_checkpoint(
    path: str,
    connector,
    optimizer=None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load Connector weights plus optimizer state for training resume."""
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    connector.load_state_dict(checkpoint["connector"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint