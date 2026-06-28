from __future__ import annotations

import math

import torch
from torch import nn


class SigLIPVisionEncoder(nn.Module):
    """Frozen SigLIP2 vision encoder that returns patch tokens."""

    def __init__(
        self,
        model_name: str = "google/siglip2-large-patch16-256",
        image_size: int = 256,
        patch_size: int = 16,
        torch_dtype=None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.processor = AutoProcessor.from_pretrained(model_name)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModel.from_pretrained(model_name, **kwargs)

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @property
    def grid_size(self) -> tuple[int, int]:
        grid = self.image_size // self.patch_size
        return grid, grid

    def _vision_forward(self, pixel_values: torch.Tensor):
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
        return self.model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        outputs = self._vision_forward(pixel_values)
        visual_tokens = getattr(outputs, "last_hidden_state", None)
        if visual_tokens is None and isinstance(outputs, (tuple, list)):
            visual_tokens = outputs[0]
        if visual_tokens is None:
            raise RuntimeError("SigLIP output does not expose patch tokens.")

        grid_h, grid_w = self.grid_size
        expected_tokens = grid_h * grid_w
        token_count = visual_tokens.shape[1]

        if token_count == expected_tokens + 1:
            visual_tokens = visual_tokens[:, 1:, :]
        elif token_count != expected_tokens:
            side = int(math.sqrt(token_count))
            if side * side != token_count:
                raise ValueError(
                    f"Expected {expected_tokens} patch tokens, got {token_count}."
                )
            grid_h = side
            grid_w = side

        return visual_tokens, grid_h, grid_w

