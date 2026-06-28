from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from vicaption.utils.checkpoint import save_checkpoint
from vicaption.utils.logger import get_logger


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _save_connector_checkpoint(path, model, optimizer, epoch, best_val_loss, config) -> None:
    save_checkpoint(path, model.connector, optimizer, epoch, best_val_loss, config)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        config,
        device,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        save_checkpoint_fn: Callable | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = get_logger()
        self.start_epoch = start_epoch
        self.save_checkpoint_fn = save_checkpoint_fn or _save_connector_checkpoint

        train_cfg = config["train"]
        self.grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        self.log_every = int(train_cfg.get("log_every", 20))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0) or 0.0)
        self.early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0) or 0)
        self.early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0) or 0.0)
        self.save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_name = str(train_cfg.get("best_checkpoint_name", "connector_best.pt"))
        self.last_checkpoint_name = str(train_cfg.get("last_checkpoint_name", "connector_last.pt"))
        self.use_amp = (
            device.type == "cuda"
            and str(train_cfg.get("mixed_precision", "")).lower() == "fp16"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_val_loss = best_val_loss

    def _autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _decoder_has_trainable_parameters(self) -> bool:
        return any(parameter.requires_grad for parameter in self.model.decoder.parameters())

    def _set_train_mode(self) -> None:
        self.model.connector.train()
        self.model.vision_encoder.eval()
        if self._decoder_has_trainable_parameters():
            self.model.decoder.train()
        else:
            self.model.decoder.eval()

    def train_one_epoch(self, epoch: int) -> float:
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch - 1)
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        num_steps = 0

        progress = tqdm(self.train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, self.device)
            with self._autocast():
                outputs = self.model(batch)
                raw_loss = outputs["loss"]
                loss = raw_loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            should_step = (step + 1) % self.grad_accum_steps == 0 or step + 1 == len(self.train_loader)
            if should_step:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    trainable_params = [p for group in self.optimizer.param_groups for p in group["params"]]
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            loss_value = float(raw_loss.detach().cpu())
            total_loss += loss_value
            num_steps += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")
            if self.log_every and (step + 1) % self.log_every == 0:
                self.logger.info("epoch=%s step=%s loss=%.4f", epoch, step + 1, loss_value)

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        progress = tqdm(self.val_loader, desc=f"val epoch {epoch}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, self.device)
            with self._autocast():
                outputs = self.model(batch)
            loss_value = float(outputs["loss"].detach().cpu())
            total_loss += loss_value
            num_steps += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

        return total_loss / max(num_steps, 1)

    def fit(self) -> dict[str, float]:
        epochs = int(self.config["train"].get("epochs", 1))
        validate_every_epoch = bool(self.config["train"].get("validate_every_epoch", True))
        history = {"train_loss": float("inf"), "val_loss": float("inf")}

        if self.start_epoch > epochs:
            self.logger.info(
                "start_epoch=%s is greater than configured epochs=%s; nothing to train.",
                self.start_epoch,
                epochs,
            )
            return history

        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch) if validate_every_epoch else float("inf")
            history = {"train_loss": train_loss, "val_loss": val_loss}
            self.logger.info(
                "epoch=%s train_loss=%.4f val_loss=%s",
                epoch,
                train_loss,
                f"{val_loss:.4f}" if val_loss != float("inf") else "inf",
            )

            criterion = val_loss if val_loss != float("inf") else train_loss
            improved = criterion < self.best_val_loss - self.early_stopping_min_delta
            if improved:
                self.best_val_loss = criterion
                epochs_without_improvement = 0
                best_path = self.save_dir / self.best_checkpoint_name
                self.save_checkpoint_fn(best_path, self.model, self.optimizer, epoch, self.best_val_loss, self.config)
            else:
                epochs_without_improvement += 1

            last_path = self.save_dir / self.last_checkpoint_name
            self.save_checkpoint_fn(last_path, self.model, self.optimizer, epoch, self.best_val_loss, self.config)

            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    "early stopping after %s epochs without improvement; best_val_loss=%.4f",
                    epochs_without_improvement,
                    self.best_val_loss,
                )
                break

        return history