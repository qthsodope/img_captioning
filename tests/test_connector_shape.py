from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.models.connector import QwenStyleConnector


def test_connector_shape():
    connector = QwenStyleConnector(vision_dim=1024, llm_dim=2048, spatial_merge_size=2)
    visual_tokens = torch.randn(2, 256, 1024)

    output = connector(visual_tokens, grid_h=16, grid_w=16)

    assert output.shape == (2, 64, 2048)


def test_connector_rejects_mismatched_grid():
    connector = QwenStyleConnector(vision_dim=1024, llm_dim=2048, spatial_merge_size=2)
    visual_tokens = torch.randn(2, 255, 1024)

    with pytest.raises(AssertionError):
        connector(visual_tokens, grid_h=16, grid_w=16)

