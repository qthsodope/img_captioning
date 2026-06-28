from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.train.train_lora_connector import run_lora_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_lora.yaml")
    args = parser.parse_args()

    history = run_lora_training(args.config)
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()