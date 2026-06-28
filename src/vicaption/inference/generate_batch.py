from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from vicaption.inference.generate_one import generate_caption
from vicaption.models.captioner import VietnameseCaptioner
from vicaption.models.connector import QwenStyleConnector
from vicaption.models.decoder import QwenDecoder, decoder_kwargs_from_config
from vicaption.models.vision_encoder import SigLIPVisionEncoder
from vicaption.utils.checkpoint import load_connector_checkpoint, load_lora_connector_checkpoint
from vicaption.utils.config import load_config
from vicaption.utils.device import get_device


def _move_module_to_device(module, device) -> None:
    try:
        module.to(device)
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "quant" not in message and "bitsandbytes" not in message and "4-bit" not in message:
            raise


def load_generation_items(json_path: str | Path) -> list[dict]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Generation input JSON must be a list.")
    return data


def save_predictions(predictions: list[dict[str, str]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
        file.write("\n")


def generate_batch_predictions(
    model,
    processor,
    tokenizer,
    items: list[dict],
    image_dir: str | Path,
    prompt: str,
    generation_config: dict,
    device,
) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    seen: set[str] = set()
    image_root = Path(image_dir)

    for item in tqdm(items, desc="generate", leave=False):
        image_id = item["image_id"]
        if image_id in seen:
            continue
        prediction = generate_caption(
            image_path=str(image_root / image_id),
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            prompt=prompt,
            generation_config=generation_config,
            device=device,
        )
        predictions.append({"image_id": image_id, "prediction": prediction})
        seen.add(image_id)

    return predictions


def _enable_lora_from_config(decoder: QwenDecoder, config: dict) -> None:
    lora_cfg = config.get("lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return
    decoder.enable_lora(
        r=int(lora_cfg.get("r", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        gradient_checkpointing=False,
    )


def run_batch_generation(config_path: str) -> list[dict[str, str]]:
    config = load_config(config_path)
    device = get_device(config["project"].get("device", "cuda"))
    model_cfg = config["model"]
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    vision_encoder = SigLIPVisionEncoder(
        model_name=model_cfg["vision_encoder"],
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        torch_dtype=dtype,
    )
    decoder = QwenDecoder(
        model_name=model_cfg["decoder"],
        torch_dtype=dtype,
        **decoder_kwargs_from_config(model_cfg),
    )
    _enable_lora_from_config(decoder, config)

    connector = QwenStyleConnector(
        vision_dim=int(model_cfg["vision_dim"]),
        llm_dim=int(model_cfg["llm_dim"]),
        spatial_merge_size=int(model_cfg["spatial_merge_size"]),
    )
    if bool(config.get("lora", {}).get("enabled", False)):
        load_lora_connector_checkpoint(model_cfg["checkpoint"], connector, decoder, map_location="cpu")
    else:
        load_connector_checkpoint(model_cfg["checkpoint"], connector, map_location=str(device))

    model = VietnameseCaptioner(vision_encoder, connector, decoder)
    _move_module_to_device(model.vision_encoder, device)
    _move_module_to_device(model.connector, device)
    _move_module_to_device(model.decoder, device)
    model.eval()

    items = load_generation_items(config["data"]["test_json"])
    predictions = generate_batch_predictions(
        model=model,
        processor=vision_encoder.processor,
        tokenizer=decoder.tokenizer,
        items=items,
        image_dir=config["data"]["image_dir"],
        prompt=config["prompt"]["text"],
        generation_config=config["generation"],
        device=device,
    )
    save_predictions(predictions, config["outputs"]["predictions"])
    return predictions