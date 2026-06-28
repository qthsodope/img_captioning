from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


def _resolve_torch_dtype(value: str | torch.dtype | None, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value

    normalized = str(value).lower()
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return aliases[normalized]


def _normalize_optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


def decoder_kwargs_from_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract optional QwenDecoder loading options from model config."""
    return {
        "load_in_4bit": bool(model_cfg.get("decoder_load_in_4bit", False)),
        "bnb_4bit_compute_dtype": model_cfg.get("decoder_4bit_compute_dtype"),
        "bnb_4bit_quant_type": str(model_cfg.get("decoder_4bit_quant_type", "nf4")),
        "bnb_4bit_use_double_quant": bool(model_cfg.get("decoder_4bit_use_double_quant", True)),
        "device_map": _normalize_optional(model_cfg.get("decoder_device_map")),
        "attn_implementation": _normalize_optional(model_cfg.get("decoder_attn_implementation")),
    }


class QwenDecoder(nn.Module):
    """Qwen causal decoder that accepts multimodal input embeddings."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: str | torch.dtype | None = None,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        device_map: str | dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        low_cpu_mem_usage: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.lora_enabled = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError("bitsandbytes/transformers quantization support is required for 4-bit Qwen.") from exc

            compute_dtype = _resolve_torch_dtype(bnb_4bit_compute_dtype, torch_dtype)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
            if device_map is not None:
                model_kwargs["device_map"] = device_map
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            if device_map is not None:
                model_kwargs["device_map"] = device_map

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except (TypeError, ValueError) as exc:
            message = str(exc).lower()
            if "attn" not in message and "sdpa" not in message:
                raise
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.config.use_cache = False

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def enable_lora(
        self,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
        target_modules: Sequence[str] | None = None,
        gradient_checkpointing: bool = True,
    ) -> None:
        """Attach PEFT LoRA adapters to Qwen attention projections."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("peft is required for LoRA training. Install it with `pip install peft`.") from exc

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if self.load_in_4bit:
            try:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=gradient_checkpointing,
                )
            except TypeError:
                self.model = prepare_model_for_kbit_training(self.model)
        elif gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        modules = list(target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"])
        config = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, config)
        self.model.config.use_cache = False
        self.lora_enabled = True

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        if not self.lora_enabled:
            return {}
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(self.model)

    def load_lora_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        if not self.lora_enabled:
            raise RuntimeError("LoRA must be enabled before loading LoRA weights.")
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state_dict)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.model.get_input_embeddings()
        return embedding(input_ids.to(embedding.weight.device))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        model_device = self.model.get_input_embeddings().weight.device
        inputs_embeds = inputs_embeds.to(model_device)
        attention_mask = attention_mask.to(model_device)
        if labels is not None:
            labels = labels.to(model_device)

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs,
    ):
        model_device = self.model.get_input_embeddings().weight.device
        return self.model.generate(
            inputs_embeds=inputs_embeds.to(model_device),
            attention_mask=attention_mask.to(model_device),
            **generation_kwargs,
        )