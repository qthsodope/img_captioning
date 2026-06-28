from __future__ import annotations

import json
from pathlib import Path

from vicaption.eval.metrics import compute_all_metrics
from vicaption.utils.config import load_config


def load_predictions(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    if not isinstance(data, list):
        raise ValueError("Predictions must be a list or mapping.")
    return {item["image_id"]: item.get("prediction", "") for item in data}


def load_references(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("References must be a list.")
    return {item["image_id"]: list(item["references"]) for item in data}


def write_metrics(metrics: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_evaluation(config_path: str) -> dict:
    config = load_config(config_path)
    predictions = load_predictions(config["outputs"]["predictions"])
    references = load_references(config["data"]["test_refs_json"])
    metrics = compute_all_metrics(predictions, references)
    write_metrics(metrics, config["outputs"]["metrics"])
    return metrics

