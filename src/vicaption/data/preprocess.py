from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_raw_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load raw Flickr30k Vietnamese annotations."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Raw annotations JSON must be a list.")
    return data


def flatten_annotations(raw_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert image-level caption lists to flat image/caption samples."""
    flat: list[dict[str, str]] = []
    for item in raw_items:
        image_id = item.get("image_id")
        captions = item.get("captions")
        if not isinstance(image_id, str) or not isinstance(captions, list):
            raise ValueError("Each raw item must have image_id and captions.")

        for caption in captions:
            if not isinstance(caption, str):
                raise ValueError("Each caption must be a string.")
            flat.append({"image_id": image_id, "caption": caption})

    return flat


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with UTF-8 Vietnamese text preserved."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

