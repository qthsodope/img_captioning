from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.data.validate import validate_raw_annotations
from vicaption.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    output_path = Path(data_cfg["stats_dir"]) / "data_validation.json"
    stats = validate_raw_annotations(data_cfg["raw_json"], data_cfg["image_dir"], output_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

