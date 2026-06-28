from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any

from vicaption.data.preprocess import flatten_annotations


def split_by_image_id(
    raw_items: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Split raw items by image_id so the same image never crosses splits."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in raw_items:
        image_id = item.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("Each raw item must have a non-empty image_id.")
        grouped.setdefault(image_id, []).append(item)

    image_ids = list(grouped.keys())
    random.Random(seed).shuffle(image_ids)

    total = len(image_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_ids = {
        "train": set(image_ids[:train_end]),
        "val": set(image_ids[train_end:val_end]),
        "test": set(image_ids[val_end:]),
    }

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for name, ids in split_ids.items():
        for image_id in image_ids:
            if image_id in ids:
                splits[name].extend(grouped[image_id])
    return splits


def build_flat_split(raw_split_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build flat image/caption samples for a raw split."""
    return flatten_annotations(raw_split_items)


def build_refs(raw_split_items: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    """Build reference captions per image_id."""
    refs: list[dict[str, list[str]]] = []
    seen: set[str] = set()
    for item in raw_split_items:
        image_id = item["image_id"]
        if image_id in seen:
            continue
        captions = item.get("captions", [])
        if not isinstance(captions, list):
            raise ValueError("Each raw split item must have captions.")
        refs.append({"image_id": image_id, "references": list(captions)})
        seen.add(image_id)
    return refs


def split_stats(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return compact split statistics."""
    stats: dict[str, Any] = {}
    for name, items in splits.items():
        image_ids = {item["image_id"] for item in items}
        stats[name] = {
            "num_images": len(image_ids),
            "num_flat_samples": len(build_flat_split(items)),
        }
    return stats

