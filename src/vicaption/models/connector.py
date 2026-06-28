from __future__ import annotations

import torch
from torch import nn


class QwenStyleConnector(nn.Module):
    """Qwen-style spatial merge connector from SigLIP patch tokens to Qwen hidden dim."""

    def __init__(
        self,
        vision_dim: int = 1024,
        llm_dim: int = 2048,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.spatial_merge_size = spatial_merge_size
        merged_dim = vision_dim * spatial_merge_size * spatial_merge_size

        self.layer_norm = nn.LayerNorm(vision_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, merged_dim),
            nn.GELU(),
            nn.Linear(merged_dim, llm_dim),
        )

    def spatial_merge(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Merge each spatial_merge_size x spatial_merge_size block into one token."""
        assert x.shape[-1] == self.vision_dim
        assert grid_h * grid_w == x.shape[1]
        assert grid_h % self.spatial_merge_size == 0
        assert grid_w % self.spatial_merge_size == 0

        batch_size, _, channels = x.shape
        merge = self.spatial_merge_size
        x = x.view(batch_size, grid_h, grid_w, channels)
        x = x.view(batch_size, grid_h // merge, merge, grid_w // merge, merge, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, (grid_h // merge) * (grid_w // merge), merge * merge * channels)

    def forward(self, visual_tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        assert visual_tokens.shape[-1] == self.vision_dim
        assert grid_h * grid_w == visual_tokens.shape[1]
        assert grid_h % self.spatial_merge_size == 0
        assert grid_w % self.spatial_merge_size == 0

        visual_tokens = self.layer_norm(visual_tokens)
        merged = self.spatial_merge(visual_tokens, grid_h, grid_w)
        return self.mlp(merged)

