from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.inference.generate_batch import generate_batch_predictions, save_predictions
from vicaption.inference.generate_one import generate_caption


class FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.ones(len(texts), 2, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 2, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "numel") and ids.numel() == 0:
            return ""
        return "First sentence. Second sentence."


class FakeModel:
    def __init__(self, empty=False):
        self.empty = empty

    def generate(self, pixel_values, prompt_input_ids, prompt_attention_mask, generation_kwargs):
        if self.empty:
            return torch.empty(1, 0, dtype=torch.long)
        return torch.tensor([[1, 2, 3]])


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)


def test_generate_caption_returns_string(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path)

    caption = generate_caption(
        str(image_path),
        model=FakeModel(),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        prompt="prompt",
        generation_config={},
        device=torch.device("cpu"),
    )

    assert isinstance(caption, str)
    assert caption == "First sentence."


def test_generate_caption_can_return_raw_text_without_postprocess(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path)

    caption = generate_caption(
        str(image_path),
        model=FakeModel(),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        prompt="prompt",
        generation_config={"postprocess": False},
        device=torch.device("cpu"),
    )

    assert caption == "First sentence. Second sentence."


def test_empty_prediction_is_handled(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path)

    caption = generate_caption(
        str(image_path),
        model=FakeModel(empty=True),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        prompt="prompt",
        generation_config={},
        device=torch.device("cpu"),
    )

    assert caption == ""


def test_batch_generation_saves_valid_json(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_image(image_dir / "sample.jpg")

    predictions = generate_batch_predictions(
        model=FakeModel(),
        processor=FakeProcessor(),
        tokenizer=FakeTokenizer(),
        items=[{"image_id": "sample.jpg", "caption": "c1"}],
        image_dir=image_dir,
        prompt="prompt",
        generation_config={},
        device=torch.device("cpu"),
    )
    output_path = tmp_path / "predictions.json"
    save_predictions(predictions, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == [{"image_id": "sample.jpg", "prediction": "First sentence."}]
