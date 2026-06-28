from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vicaption.data.collator import CaptionCollator
from vicaption.data.dataset import Flickr30kViDataset, LimitedDataset, OneCaptionPerImageDataset
from vicaption.data.feature_cache import (
    collect_unique_image_ids,
    feature_cache_path,
    is_valid_feature_cache,
    precompute_vision_features,
)
from vicaption.models.captioner import VietnameseCaptioner
from vicaption.models.connector import QwenStyleConnector
from vicaption.models.decoder import QwenDecoder, decoder_kwargs_from_config
from vicaption.models.vision_encoder import SigLIPVisionEncoder
from vicaption.train.train_connector import (
    CachedVisionEncoderStub,
    _apply_optimizer_hparams_from_config,
    _move_module_to_device,
    _move_optimizer_to_device,
)
from vicaption.train.trainer import Trainer
from vicaption.utils.checkpoint import (
    load_connector_checkpoint,
    load_lora_connector_checkpoint,
    save_lora_connector_checkpoint,
)
from vicaption.utils.config import load_config
from vicaption.utils.device import get_device
from vicaption.utils.seed import set_seed


def _optimizer_param_groups(model, train_cfg: dict) -> list[dict]:
    connector_params = [p for p in model.connector.parameters() if p.requires_grad]
    lora_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if not connector_params:
        raise RuntimeError("Connector has no trainable parameters.")
    if not lora_params:
        raise RuntimeError("Decoder LoRA has no trainable parameters.")

    connector_lr = float(train_cfg.get("connector_lr", train_cfg.get("lr", 1e-5)))
    lora_lr = float(train_cfg.get("lora_lr", train_cfg.get("lr", 5e-5)))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    return [
        {"params": connector_params, "lr": connector_lr, "weight_decay": weight_decay, "name": "connector"},
        {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay, "name": "decoder_lora"},
    ]


def _resolve_lora_resume_path(train_cfg: dict, save_dir: Path) -> Path | None:
    resume_value = train_cfg.get("resume_from_checkpoint")
    if resume_value:
        resume_path = Path(resume_value)
        return resume_path if resume_path.is_absolute() else Path(resume_value)
    if bool(train_cfg.get("auto_resume", True)):
        candidate = save_dir / str(train_cfg.get("last_checkpoint_name", "lora_connector_last.pt"))
        if candidate.exists():
            return candidate
    return None


def _save_lora_checkpoint(path, model, optimizer, epoch, best_val_loss, config) -> None:
    save_lora_connector_checkpoint(path, model.connector, model.decoder, optimizer, epoch, best_val_loss, config)


def _count_trainable(model) -> tuple[int, int, int]:
    connector = sum(p.numel() for p in model.connector.parameters() if p.requires_grad)
    decoder = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    return connector + decoder, connector, decoder


def run_lora_training(config_path: str) -> dict[str, float]:
    config = load_config(config_path)
    set_seed(int(config["project"].get("seed", 42)))
    device = get_device(config["project"].get("device", "cuda"))

    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    lora_cfg = config.get("lora", {})
    dtype = torch.float16 if device.type == "cuda" and train_cfg.get("mixed_precision") == "fp16" else torch.float32

    processed_dir = Path(data_cfg["processed_dir"])
    if bool(train_cfg.get("one_caption_per_image_per_epoch", False)):
        train_dataset = OneCaptionPerImageDataset(
            str(processed_dir / "train.json"),
            data_cfg["image_dir"],
            captions_per_image_per_epoch=int(train_cfg.get("captions_per_image_per_epoch", 1)),
            caption_order=str(train_cfg.get("caption_order", "original")),
        )
    else:
        train_dataset = Flickr30kViDataset(str(processed_dir / "train.json"), data_cfg["image_dir"])
    val_dataset = Flickr30kViDataset(str(processed_dir / "val.json"), data_cfg["image_dir"])
    val_max_samples = int(train_cfg.get("val_max_samples", 0) or 0)
    if val_max_samples > 0:
        val_dataset = LimitedDataset(val_dataset, val_max_samples)
    if len(train_dataset) == 0:
        raise ValueError("Training split is empty. Run preprocessing after adding Flickr30k Vietnamese data.")

    use_cached_features = bool(train_cfg.get("use_cached_vision_features", False))
    feature_dir = data_cfg.get("feature_dir", str(processed_dir / "vision_features"))
    vision_encoder = None
    if use_cached_features:
        image_ids = collect_unique_image_ids([train_dataset, val_dataset])
        missing_features = [
            image_id
            for image_id in image_ids
            if not is_valid_feature_cache(feature_cache_path(feature_dir, image_id))
        ]
        if missing_features and not bool(train_cfg.get("build_feature_cache", True)):
            raise FileNotFoundError(
                f"Missing {len(missing_features)} cached SigLIP features and build_feature_cache is false."
            )
        if missing_features:
            vision_encoder = SigLIPVisionEncoder(
                model_name=model_cfg["vision_encoder"],
                image_size=int(model_cfg["image_size"]),
                patch_size=int(model_cfg["patch_size"]),
                torch_dtype=dtype,
            )
            vision_encoder.to(device)
            precompute_vision_features(
                image_ids=missing_features,
                image_dir=data_cfg["image_dir"],
                feature_dir=feature_dir,
                processor=vision_encoder.processor,
                vision_encoder=vision_encoder,
                device=device,
                batch_size=int(train_cfg.get("feature_cache_batch_size", 8)),
            )
            vision_encoder.to("cpu")
            del vision_encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
        vision_encoder = CachedVisionEncoderStub()
    else:
        vision_encoder = SigLIPVisionEncoder(
            model_name=model_cfg["vision_encoder"],
            image_size=int(model_cfg["image_size"]),
            patch_size=int(model_cfg["patch_size"]),
            torch_dtype=dtype,
        )
        vision_encoder.to(device)

    decoder = QwenDecoder(
        model_name=model_cfg["decoder"],
        torch_dtype=dtype,
        **decoder_kwargs_from_config(model_cfg),
    )
    decoder.enable_lora(
        r=int(lora_cfg.get("r", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
    )

    connector = QwenStyleConnector(
        vision_dim=int(model_cfg["vision_dim"]),
        llm_dim=int(model_cfg["llm_dim"]),
        spatial_merge_size=int(model_cfg["spatial_merge_size"]),
    )
    base_connector_checkpoint = train_cfg.get("base_connector_checkpoint") or model_cfg.get("connector_checkpoint")
    if not base_connector_checkpoint:
        raise ValueError("LoRA training requires train.base_connector_checkpoint or model.connector_checkpoint.")
    load_connector_checkpoint(str(base_connector_checkpoint), connector, map_location="cpu")

    model = VietnameseCaptioner(vision_encoder, connector, decoder)
    _move_module_to_device(model.decoder, device)
    _move_module_to_device(model.connector, device)
    if use_cached_features:
        _move_module_to_device(model.vision_encoder, "cpu")
    else:
        _move_module_to_device(model.vision_encoder, device)

    total_trainable, connector_trainable, decoder_trainable = _count_trainable(model)
    print(
        "trainable parameters: "
        f"total={total_trainable:,} connector={connector_trainable:,} decoder_lora={decoder_trainable:,}"
    )

    prompt_cfg = config.get("prompt", {})
    base_prompt = prompt_cfg["text"]
    train_prompts = (
        prompt_cfg.get("train_texts_by_style")
        or prompt_cfg.get("train_texts")
        or prompt_cfg.get("texts")
        or base_prompt
    )
    common_collator_kwargs = dict(
        siglip_processor=getattr(vision_encoder, "processor", None),
        qwen_tokenizer=decoder.tokenizer,
        max_prompt_length=int(train_cfg["max_prompt_length"]),
        max_caption_length=int(train_cfg["max_caption_length"]),
        feature_dir=feature_dir if use_cached_features else None,
        append_eos_to_caption=bool(train_cfg.get("append_eos_to_caption", False)),
        normalize_caption_text=bool(train_cfg.get("normalize_caption_text", True)),
        single_sentence_targets=bool(train_cfg.get("single_sentence_targets", False)),
        ensure_terminal_punctuation=bool(train_cfg.get("ensure_terminal_punctuation", False)),
        concise_max_words=int(train_cfg.get("concise_max_words", 9)),
        detailed_min_words=int(train_cfg.get("detailed_min_words", 14)),
    )
    train_collator = CaptionCollator(
        prompt=train_prompts,
        prompt_selection=str(train_cfg.get("prompt_selection", "hash")),
        **common_collator_kwargs,
    )
    val_collator = CaptionCollator(
        prompt=base_prompt,
        prompt_selection="first",
        **common_collator_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=train_collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(train_cfg["num_workers"]),
            collate_fn=val_collator,
            pin_memory=device.type == "cuda",
        )

    optimizer = torch.optim.AdamW(_optimizer_param_groups(model, train_cfg))

    start_epoch = 1
    best_val_loss = float("inf")
    save_dir = Path(train_cfg.get("save_dir", "checkpoints_lora"))
    resume_path = _resolve_lora_resume_path(train_cfg, save_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_lora_connector_checkpoint(
            str(resume_path),
            model.connector,
            model.decoder,
            optimizer=optimizer,
            map_location="cpu",
        )
        _move_module_to_device(model.connector, device)
        _move_optimizer_to_device(optimizer, device)
        if bool(train_cfg.get("override_optimizer_hparams_on_resume", False)):
            _apply_optimizer_hparams_from_config(optimizer, train_cfg)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        if bool(train_cfg.get("reset_best_on_resume", False)) or not math.isfinite(best_val_loss):
            best_val_loss = float("inf")
        print(
            f"Resuming LoRA from {resume_path}: "
            f"last_epoch={checkpoint.get('epoch', 0)}, start_epoch={start_epoch}, "
            f"best_val_loss={best_val_loss}"
        )

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        config,
        device,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        save_checkpoint_fn=_save_lora_checkpoint,
    )
    return trainer.fit()
