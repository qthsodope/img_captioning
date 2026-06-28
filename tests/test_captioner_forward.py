from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.models.captioner import VietnameseCaptioner
from vicaption.models.connector import QwenStyleConnector


class FakeVisionEncoder(nn.Module):
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        tokens = torch.arange(batch_size * 4 * 3, dtype=torch.float32).view(batch_size, 4, 3)
        return tokens, 2, 2


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 5)
        self.lm_head = nn.Linear(5, 16)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def embed_tokens(self, input_ids):
        return self.embedding(input_ids)

    def forward(self, inputs_embeds, attention_mask, labels=None):
        causal_context = inputs_embeds.cumsum(dim=1)
        logits = self.lm_head(causal_context)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, inputs_embeds, attention_mask, **generation_kwargs):
        return torch.tensor([[1, 2, 3]])


def _build_model():
    connector = QwenStyleConnector(vision_dim=3, llm_dim=5, spatial_merge_size=1)
    return VietnameseCaptioner(FakeVisionEncoder(), connector, FakeDecoder())


def test_captioner_builds_caption_only_labels():
    model = _build_model()
    batch = {
        "prompt_input_ids": torch.tensor([[1, 2]]),
        "prompt_attention_mask": torch.tensor([[1, 1]]),
        "caption_input_ids": torch.tensor([[3, 4, 0]]),
        "caption_attention_mask": torch.tensor([[1, 1, 0]]),
    }
    visual_embeds = torch.zeros(1, 4, 5)
    built = model.build_multimodal_inputs(
        prompt_embeds=model.decoder.embed_tokens(batch["prompt_input_ids"]),
        visual_embeds=visual_embeds,
        caption_embeds=model.decoder.embed_tokens(batch["caption_input_ids"]),
        prompt_attention_mask=batch["prompt_attention_mask"],
        caption_input_ids=batch["caption_input_ids"],
        caption_attention_mask=batch["caption_attention_mask"],
    )

    assert built["inputs_embeds"].shape[1] == 9
    assert built["attention_mask"].shape == (1, 9)
    assert built["labels"].shape == (1, 9)
    assert torch.all(built["labels"][:, :6] == -100)
    assert built["labels"][0, 6:].tolist() == [3, 4, -100]


def test_captioner_forward_returns_scalar_loss_and_connector_gradients():
    model = _build_model()
    batch = {
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "prompt_input_ids": torch.tensor([[1, 2]]),
        "prompt_attention_mask": torch.tensor([[1, 1]]),
        "caption_input_ids": torch.tensor([[3, 4, 5]]),
        "caption_attention_mask": torch.tensor([[1, 1, 1]]),
    }

    outputs = model(batch)
    assert outputs["loss"].ndim == 0
    outputs["loss"].backward()

    connector_grads = [p.grad for p in model.connector.parameters() if p.requires_grad]
    assert connector_grads
    assert any(grad is not None and torch.any(grad != 0) for grad in connector_grads)
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
