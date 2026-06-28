from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image
import torch

from vicaption.data.feature_cache import feature_cache_path, load_feature_cache


class CaptionCollator:
    """Load images and tokenize prompt/captions for captioner training."""

    def __init__(
        self,
        siglip_processor,
        qwen_tokenizer,
        prompt: str | Sequence[str] | Mapping[str, str | Sequence[str]],
        max_prompt_length: int,
        max_caption_length: int,
        feature_dir: str | None = None,
        append_eos_to_caption: bool = False,
        normalize_caption_text: bool = True,
        single_sentence_targets: bool = False,
        ensure_terminal_punctuation: bool = False,
        prompt_selection: str = "hash",
        concise_max_words: int = 9,
        detailed_min_words: int = 14,
    ):
        self.siglip_processor = siglip_processor
        self.qwen_tokenizer = qwen_tokenizer
        self.prompts_by_style: dict[str, list[str]] = {}
        if isinstance(prompt, Mapping):
            for style, value in prompt.items():
                if isinstance(value, str):
                    prompts = [value]
                else:
                    prompts = [item for item in value if item]
                if prompts:
                    self.prompts_by_style[str(style)] = prompts
            self.prompts = [item for prompts in self.prompts_by_style.values() for item in prompts]
        elif isinstance(prompt, str):
            self.prompts = [prompt]
        else:
            self.prompts = [item for item in prompt if item]
        if not self.prompts:
            raise ValueError("At least one prompt is required.")
        self.prompt = self.prompts[0]
        self.max_prompt_length = max_prompt_length
        self.max_caption_length = max_caption_length
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.append_eos_to_caption = append_eos_to_caption
        self.normalize_caption_text = normalize_caption_text
        self.single_sentence_targets = single_sentence_targets
        self.ensure_terminal_punctuation = ensure_terminal_punctuation
        self.prompt_selection = prompt_selection
        self.concise_max_words = int(concise_max_words)
        self.detailed_min_words = int(detailed_min_words)

    def _prepare_caption_text(self, caption: str) -> str:
        text = caption.strip()
        if self.normalize_caption_text:
            text = re.sub(r"\s+", " ", text)
        if self.single_sentence_targets:
            match = re.search(r"(.+?[.!?])(?:\s|$)", text)
            if match:
                text = match.group(1).strip()
        if self.ensure_terminal_punctuation and text and text[-1] not in ".!?":
            text = text.rstrip(",;: ")
            text += "."
        return text

    def _tokenize_captions(self, captions: list[str]) -> dict:
        eos_token_id = getattr(self.qwen_tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.qwen_tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        append_eos = self.append_eos_to_caption and eos_token_id is not None
        tokenize_max_length = self.max_caption_length - 1 if append_eos else self.max_caption_length
        tokenize_max_length = max(1, tokenize_max_length)

        encoded = self.qwen_tokenizer(
            captions,
            padding=False,
            truncation=True,
            max_length=tokenize_max_length,
            add_special_tokens=False,
        )
        token_rows = encoded["input_ids"]
        if hasattr(token_rows, "tolist"):
            token_rows = token_rows.tolist()

        input_rows: list[list[int]] = []
        for row in token_rows:
            token_ids = [int(token_id) for token_id in row]
            if append_eos and (not token_ids or token_ids[-1] != eos_token_id):
                token_ids.append(int(eos_token_id))
            if len(token_ids) > self.max_caption_length:
                token_ids = token_ids[: self.max_caption_length]
                if append_eos:
                    token_ids[-1] = int(eos_token_id)
            input_rows.append(token_ids)

        padded_length = max(1, max((len(row) for row in input_rows), default=0))
        input_ids = []
        attention_mask = []
        for row in input_rows:
            pad_count = padded_length - len(row)
            input_ids.append(row + [int(pad_token_id)] * pad_count)
            attention_mask.append([1] * len(row) + [0] * pad_count)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _select_prompt(self, item: dict[str, str], caption: str) -> str:
        if len(self.prompts) == 1 or self.prompt_selection == "first":
            return self.prompts[0]
        if self.prompt_selection == "length_bucket":
            word_count = len(caption.split())
            if word_count <= self.concise_max_words:
                candidates = self.prompts_by_style.get("concise", self.prompts)
            elif word_count >= self.detailed_min_words:
                candidates = self.prompts_by_style.get("detailed", self.prompts)
            else:
                candidates = self.prompts_by_style.get("balanced", self.prompts)
            return self._select_hashed_prompt(candidates, item, caption)
        if self.prompt_selection != "hash":
            raise ValueError(f"Unsupported prompt_selection: {self.prompt_selection}")

        return self._select_hashed_prompt(self.prompts, item, caption)

    @staticmethod
    def _select_hashed_prompt(candidates: list[str], item: dict[str, str], caption: str) -> str:
        key = f"{item.get('image_id', '')}\0{caption}".encode("utf-8")
        index = int(hashlib.sha1(key).hexdigest(), 16) % len(candidates)
        return candidates[index]

    def __call__(self, batch: list[dict[str, str]]) -> dict:
        if self.feature_dir is None:
            images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
            image_batch = self.siglip_processor(images=images, return_tensors="pt")
        else:
            feature_items = []
            for item in batch:
                cache_path = feature_cache_path(self.feature_dir, item["image_id"])
                feature_items.append(load_feature_cache(cache_path))
            visual_tokens = torch.stack([item["visual_tokens"] for item in feature_items], dim=0)
            image_batch = {
                "visual_tokens": visual_tokens,
                "grid_h": int(feature_items[0]["grid_h"]),
                "grid_w": int(feature_items[0]["grid_w"]),
            }

        image_ids = [item["image_id"] for item in batch]
        captions = [self._prepare_caption_text(item["caption"]) for item in batch]
        prompts = [self._select_prompt(item, caption) for item, caption in zip(batch, captions)]

        prompt_batch = self.qwen_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        caption_batch = self._tokenize_captions(captions)

        output = {
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "caption_input_ids": caption_batch["input_ids"],
            "caption_attention_mask": caption_batch["attention_mask"],
            "image_ids": image_ids,
            "captions": captions,
            "prompts": prompts,
        }
        output.update(image_batch)
        return output
