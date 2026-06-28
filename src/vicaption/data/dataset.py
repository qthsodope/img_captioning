from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - keeps lightweight data tests importable
    class Dataset:  # type: ignore[no-redef]
        pass


class Flickr30kViDataset(Dataset):
    """Flat Flickr30k Vietnamese caption dataset."""

    def __init__(self, json_path: str, image_dir: str):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list.")
        self.items: list[dict[str, Any]] = data

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, str]:
        item = self.items[index]
        image_id = item.get("image_id")
        caption = item.get("caption")
        if not isinstance(image_id, str) or not isinstance(caption, str):
            raise ValueError("Each dataset item must contain image_id and caption strings.")

        return {
            "image_id": image_id,
            "image_path": str(self.image_dir / image_id),
            "caption": caption,
        }


class OneCaptionPerImageDataset(Dataset):
    """Use a small rotating caption subset for each image on every epoch."""

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        captions_per_image_per_epoch: int = 1,
        caption_order: str = "original",
    ):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.epoch = 0
        self.captions_per_image_per_epoch = max(1, int(captions_per_image_per_epoch))
        self.caption_order = caption_order

        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list.")

        grouped: OrderedDict[str, list[str]] = OrderedDict()
        for item in data:
            image_id = item.get("image_id")
            caption = item.get("caption")
            if not isinstance(image_id, str) or not isinstance(caption, str):
                raise ValueError("Each dataset item must contain image_id and caption strings.")
            grouped.setdefault(image_id, []).append(caption)

        normalized_order = caption_order.lower()
        if normalized_order in {"longest_first", "length_desc"}:
            for image_id, captions in grouped.items():
                grouped[image_id] = sorted(captions, key=lambda caption: len(caption.split()), reverse=True)
        elif normalized_order in {"length_interleave", "interleave_length"}:
            for image_id, captions in grouped.items():
                sorted_captions = sorted(captions, key=lambda caption: len(caption.split()), reverse=True)
                interleaved: list[str] = []
                left = 0
                right = len(sorted_captions) - 1
                while left <= right:
                    interleaved.append(sorted_captions[left])
                    left += 1
                    if left <= right:
                        interleaved.append(sorted_captions[right])
                        right -= 1
                grouped[image_id] = interleaved
        elif normalized_order not in {"original", "none"}:
            raise ValueError(f"Unsupported caption_order: {caption_order}")

        self.caption_groups = grouped
        self.image_ids = list(grouped.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.image_ids) * self.captions_per_image_per_epoch

    def __getitem__(self, index: int) -> dict[str, str]:
        image_index = index // self.captions_per_image_per_epoch
        caption_slot = index % self.captions_per_image_per_epoch
        image_id = self.image_ids[image_index]
        captions = self.caption_groups[image_id]
        caption_index = (self.epoch * self.captions_per_image_per_epoch + caption_slot) % len(captions)
        caption = captions[caption_index]
        return {
            "image_id": image_id,
            "image_path": str(self.image_dir / image_id),
            "caption": caption,
        }


class LimitedDataset(Dataset):
    """A lightweight deterministic view over the first max_samples of a dataset."""

    def __init__(self, dataset, max_samples: int | None):
        self.dataset = dataset
        self.max_samples = max_samples

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    @property
    def items(self):
        return getattr(self.dataset, "items", [])

    @property
    def image_ids(self):
        ids = getattr(self.dataset, "image_ids", None)
        if ids is None:
            ids = [item["image_id"] for item in self.items]
        return ids[: len(self)]

    def __len__(self) -> int:
        base_len = len(self.dataset)
        if self.max_samples is None or self.max_samples <= 0:
            return base_len
        return min(base_len, self.max_samples)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(index)
        return self.dataset[index]
