from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.collator import CaptionCollator


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __call__(
        self,
        texts,
        padding=False,
        truncation=True,
        max_length=None,
        add_special_tokens=False,
        **kwargs,
    ):
        rows = []
        for text in texts:
            token_count = len(text.split())
            row = list(range(1, token_count + 1))
            if truncation and max_length is not None:
                row = row[:max_length]
            rows.append(row)
        return {"input_ids": rows}


def _build_collator(**kwargs):
    return CaptionCollator(
        siglip_processor=None,
        qwen_tokenizer=FakeTokenizer(),
        prompt="prompt",
        max_prompt_length=8,
        max_caption_length=8,
        append_eos_to_caption=True,
        **kwargs,
    )


def test_caption_eos_is_appended_without_replacing_last_token():
    collator = _build_collator()

    batch = collator._tokenize_captions(["mot hai ba"])

    assert batch["input_ids"].tolist() == [[1, 2, 3, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]


def test_caption_text_can_be_normalized_to_single_sentence():
    collator = _build_collator(single_sentence_targets=True, ensure_terminal_punctuation=True)

    caption = collator._prepare_caption_text("  Mot nguoi dang chay. Cau thu hai.  ")
    no_punctuation = collator._prepare_caption_text("Mot nguoi dang chay")

    assert caption == "Mot nguoi dang chay."
    assert no_punctuation == "Mot nguoi dang chay."


def test_prompt_pool_selection_is_stable_for_same_item_and_caption():
    collator = CaptionCollator(
        siglip_processor=None,
        qwen_tokenizer=FakeTokenizer(),
        prompt=["prompt a", "prompt b", "prompt c"],
        max_prompt_length=8,
        max_caption_length=8,
        prompt_selection="hash",
    )
    item = {"image_id": "sample.jpg", "caption": "caption"}

    first = collator._select_prompt(item, "caption")
    second = collator._select_prompt(item, "caption")

    assert first == second
    assert first in {"prompt a", "prompt b", "prompt c"}


def test_prompt_pool_can_select_by_caption_length_bucket():
    collator = CaptionCollator(
        siglip_processor=None,
        qwen_tokenizer=FakeTokenizer(),
        prompt={
            "concise": ["short prompt"],
            "balanced": ["balanced prompt"],
            "detailed": ["detailed prompt"],
        },
        max_prompt_length=8,
        max_caption_length=8,
        prompt_selection="length_bucket",
        concise_max_words=2,
        detailed_min_words=5,
    )
    item = {"image_id": "sample.jpg", "caption": "caption"}

    assert collator._select_prompt(item, "one two") == "short prompt"
    assert collator._select_prompt(item, "one two three") == "balanced prompt"
    assert collator._select_prompt(item, "one two three four five") == "detailed prompt"
