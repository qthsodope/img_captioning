from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.preprocess import flatten_annotations, load_raw_annotations, save_json
from vicaption.data.split import build_flat_split, build_refs, split_by_image_id, split_stats
from vicaption.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    stats_dir = Path(data_cfg["stats_dir"])

    raw_items = load_raw_annotations(data_cfg["raw_json"])
    flat = flatten_annotations(raw_items)
    save_json(flat, processed_dir / "flickr30k_vi_flat.json")

    splits = split_by_image_id(
        raw_items,
        train_ratio=float(data_cfg["train_ratio"]),
        val_ratio=float(data_cfg["val_ratio"]),
        test_ratio=float(data_cfg["test_ratio"]),
        seed=int(config["project"]["seed"]),
    )
    save_json(build_flat_split(splits["train"]), processed_dir / "train.json")
    save_json(build_flat_split(splits["val"]), processed_dir / "val.json")
    save_json(build_flat_split(splits["test"]), processed_dir / "test.json")
    save_json(build_refs(splits["test"]), processed_dir / "test_refs.json")

    stats = split_stats(splits)
    save_json(stats, stats_dir / "split_stats.json")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

