from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicaption.inference.generate_batch import run_batch_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    predictions = run_batch_generation(args.config)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

