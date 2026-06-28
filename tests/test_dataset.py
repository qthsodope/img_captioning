from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.dataset import Flickr30kViDataset, LimitedDataset, OneCaptionPerImageDataset


def test_dataset_returns_expected_fields(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps([{"image_id": "sample.jpg", "caption": "Một ảnh thử."}], ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = Flickr30kViDataset(str(json_path), str(image_dir))
    item = dataset[0]

    assert len(dataset) == 1
    assert item["image_id"] == "sample.jpg"
    assert item["caption"] == "Một ảnh thử."
    assert item["image_path"] == str(image_dir / "sample.jpg")


def test_one_caption_per_image_dataset_rotates_by_epoch(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps(
            [
                {"image_id": "sample.jpg", "caption": "c1"},
                {"image_id": "sample.jpg", "caption": "c2"},
                {"image_id": "sample.jpg", "caption": "c3"},
                {"image_id": "other.jpg", "caption": "o1"},
                {"image_id": "other.jpg", "caption": "o2"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = OneCaptionPerImageDataset(str(json_path), str(image_dir))

    assert len(dataset) == 2
    assert dataset[0]["caption"] == "c1"
    dataset.set_epoch(1)
    assert dataset[0]["caption"] == "c2"
    assert dataset[1]["caption"] == "o2"


def test_one_caption_per_image_dataset_can_use_multiple_captions_each_epoch(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps(
            [
                {"image_id": "sample.jpg", "caption": "c1"},
                {"image_id": "sample.jpg", "caption": "c2"},
                {"image_id": "sample.jpg", "caption": "c3"},
                {"image_id": "other.jpg", "caption": "o1"},
                {"image_id": "other.jpg", "caption": "o2"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = OneCaptionPerImageDataset(
        str(json_path),
        str(image_dir),
        captions_per_image_per_epoch=2,
    )

    assert len(dataset) == 4
    assert [dataset[i]["caption"] for i in range(4)] == ["c1", "c2", "o1", "o2"]
    dataset.set_epoch(1)
    assert [dataset[i]["caption"] for i in range(4)] == ["c3", "c1", "o1", "o2"]


def test_one_caption_per_image_dataset_can_prioritize_long_captions(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps(
            [
                {"image_id": "sample.jpg", "caption": "short"},
                {"image_id": "sample.jpg", "caption": "a much longer caption"},
                {"image_id": "sample.jpg", "caption": "medium caption"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = OneCaptionPerImageDataset(
        str(json_path),
        str(image_dir),
        captions_per_image_per_epoch=2,
        caption_order="longest_first",
    )

    assert [dataset[i]["caption"] for i in range(2)] == ["a much longer caption", "medium caption"]


def test_one_caption_per_image_dataset_can_interleave_caption_lengths(tmp_path):
    json_path = tmp_path / "train.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps(
            [
                {"image_id": "sample.jpg", "caption": "tiny"},
                {"image_id": "sample.jpg", "caption": "a medium caption"},
                {"image_id": "sample.jpg", "caption": "the longest caption in this group"},
                {"image_id": "sample.jpg", "caption": "short caption"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = OneCaptionPerImageDataset(
        str(json_path),
        str(image_dir),
        captions_per_image_per_epoch=4,
        caption_order="length_interleave",
    )

    assert [dataset[i]["caption"] for i in range(4)] == [
        "the longest caption in this group",
        "tiny",
        "a medium caption",
        "short caption",
    ]


def test_limited_dataset_caps_length(tmp_path):
    json_path = tmp_path / "val.json"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    json_path.write_text(
        json.dumps(
            [
                {"image_id": "1.jpg", "caption": "c1"},
                {"image_id": "2.jpg", "caption": "c2"},
                {"image_id": "3.jpg", "caption": "c3"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = LimitedDataset(Flickr30kViDataset(str(json_path), str(image_dir)), max_samples=2)

    assert len(dataset) == 2
    assert dataset[1]["image_id"] == "2.jpg"
