from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList


def _move_mapping_to_device(mapping: dict, device):
    moved = {}
    for key, value in mapping.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


class StopAfterFirstSentence(StoppingCriteria):
    def __init__(self, tokenizer, min_new_tokens: int = 8):
        self.tokenizer = tokenizer
        self.min_new_tokens = min_new_tokens
        self.endings = (".", "!", "?")

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.shape[1] < self.min_new_tokens:
            return False
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
        return any(text.endswith(mark) for mark in self.endings)


class ForceEosAfterSentenceEnd(LogitsProcessor):
    """Force generation to emit EOS right after a complete sentence."""

    def __init__(self, tokenizer, min_new_tokens: int = 8):
        self.tokenizer = tokenizer
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = tokenizer.eos_token_id
        self.endings = (".", "!", "?")

    def __call__(self, input_ids, scores):
        if self.eos_token_id is None or input_ids.shape[1] < self.min_new_tokens:
            return scores

        for row in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[row], skip_special_tokens=True).strip()
            if any(text.endswith(mark) for mark in self.endings):
                scores[row, :] = -float("inf")
                scores[row, self.eos_token_id] = 0
        return scores


def postprocess_caption(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:Ảnh|Bức ảnh|Trong ảnh)\s*(?:là|cho thấy|có)?\s*", "", text, flags=re.IGNORECASE)
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    if match:
        text = match.group(1).strip()
    if text and text[-1] not in ".!?":
        text = text.rstrip(",;: ")
        text += "."
    return text


def generate_caption(
    image_path: str,
    model,
    processor,
    tokenizer,
    prompt: str,
    generation_config: dict,
    device,
) -> str:
    """Generate one Vietnamese caption string for an image."""
    image = Image.open(Path(image_path)).convert("RGB")
    pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
    prompt_batch = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=generation_config.get("max_prompt_length", 32),
        return_tensors="pt",
    )
    prompt_batch = _move_mapping_to_device(prompt_batch, device)

    generation_kwargs = dict(generation_config)
    stop_after_first_sentence = bool(generation_kwargs.pop("stop_after_first_sentence", True))
    force_eos_after_sentence = bool(generation_kwargs.pop("force_eos_after_sentence", True))
    postprocess = bool(generation_kwargs.pop("postprocess", True))
    min_new_tokens_before_stop = int(generation_kwargs.pop("min_new_tokens_before_stop", 8))
    generation_kwargs.pop("max_prompt_length", None)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    generation_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    logits_processors = LogitsProcessorList(generation_kwargs.pop("logits_processor", []))
    if force_eos_after_sentence:
        logits_processors.append(
            ForceEosAfterSentenceEnd(tokenizer, min_new_tokens=min_new_tokens_before_stop)
        )
    if logits_processors:
        generation_kwargs["logits_processor"] = logits_processors

    if stop_after_first_sentence:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopAfterFirstSentence(tokenizer, min_new_tokens=min_new_tokens_before_stop)]
        )

    sequences = model.generate(
        pixel_values=pixel_values,
        prompt_input_ids=prompt_batch["input_ids"],
        prompt_attention_mask=prompt_batch["attention_mask"],
        generation_kwargs=generation_kwargs,
    )

    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    text = tokenizer.decode(first, skip_special_tokens=True)
    if not postprocess:
        return text.strip()
    return postprocess_caption(text)
