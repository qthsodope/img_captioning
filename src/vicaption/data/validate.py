from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_raw_annotations(
    raw_json: str | Path,
    image_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Flickr30k Vietnamese raw annotations and optionally write stats."""
    raw_path = Path(raw_json)
    image_root = Path(image_dir)
    data = _load_json(raw_path)

    if not isinstance(data, list):
        raise ValueError("Raw annotations JSON must be a list.")

    image_ids: list[str] = []
    missing_images = 0
    empty_captions = 0
    invalid_items = 0

    for item in data:
        if not isinstance(item, dict):
            invalid_items += 1
            continue

        image_id = item.get("image_id")
        captions = item.get("captions")
        if not isinstance(image_id, str) or not image_id:
            invalid_items += 1
            continue
        if not isinstance(captions, list):
            invalid_items += 1
            captions = []

        image_ids.append(image_id)
        if not (image_root / image_id).exists():
            missing_images += 1

        for caption in captions:
            if not isinstance(caption, str) or not caption.strip():
                empty_captions += 1

    counts = Counter(image_ids)
    duplicate_image_ids = sum(count - 1 for count in counts.values() if count > 1)

    stats = {
        "num_items": len(data),
        "num_unique_images": len(counts),
        "num_missing_images": missing_images,
        "num_empty_captions": empty_captions,
        "num_duplicate_image_ids": duplicate_image_ids,
        "valid": (
            invalid_items == 0
            and missing_images == 0
            and empty_captions == 0
            and duplicate_image_ids == 0
        ),
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as file:
            json.dump(stats, file, ensure_ascii=False, indent=2)

    return stats

