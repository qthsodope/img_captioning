from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.preprocess import flatten_annotations


def test_flatten_annotations_preserves_image_ids_and_captions():
    raw = [
        {
            "image_id": "sample.jpg",
            "captions": ["c1", "c2", "c3", "c4", "c5"],
        }
    ]

    flat = flatten_annotations(raw)

    assert len(flat) == 5
    assert [item["image_id"] for item in flat] == ["sample.jpg"] * 5
    assert [item["caption"] for item in flat] == ["c1", "c2", "c3", "c4", "c5"]

