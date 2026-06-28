from __future__ import annotations

import torch
from torch import nn


class VietnameseCaptioner(nn.Module):
    """Compose frozen encoder, trainable connector, and frozen decoder."""

    def __init__(self, vision_encoder, connector, decoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.connector = connector
        self.decoder = decoder

    def build_multimodal_inputs(
        self,
        prompt_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        caption_input_ids: torch.Tensor,
        caption_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Concatenate embeddings and build attention mask plus caption-only labels."""
        batch_size, visual_len, _ = visual_embeds.shape
        visual_attention_mask = torch.ones(
            (batch_size, visual_len),
            dtype=prompt_attention_mask.dtype,
            device=visual_embeds.device,
        )

        inputs_embeds = torch.cat([prompt_embeds, visual_embeds, caption_embeds], dim=1)
        attention_mask = torch.cat(
            [
                prompt_attention_mask.to(visual_embeds.device),
                visual_attention_mask,
                caption_attention_mask.to(visual_embeds.device),
            ],
            dim=1,
        )

        ignore_prompt = torch.full(
            prompt_attention_mask.shape,
            -100,
            dtype=caption_input_ids.dtype,
            device=visual_embeds.device,
        )
        ignore_visual = torch.full(
            (batch_size, visual_len),
            -100,
            dtype=caption_input_ids.dtype,
            device=visual_embeds.device,
        )
        caption_labels = caption_input_ids.to(visual_embeds.device).clone()
        caption_mask = caption_attention_mask.to(visual_embeds.device)
        caption_labels = caption_labels.masked_fill(caption_mask == 0, -100)
        labels = torch.cat([ignore_prompt, ignore_visual, caption_labels], dim=1)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if "visual_tokens" in batch:
            visual_tokens = batch["visual_tokens"]
            grid_h = int(batch["grid_h"])
            grid_w = int(batch["grid_w"])
        else:
            with torch.no_grad():
                visual_tokens, grid_h, grid_w = self.vision_encoder(batch["pixel_values"])

        connector_parameter = next(self.connector.parameters())
        visual_tokens = visual_tokens.to(device=connector_parameter.device, dtype=connector_parameter.dtype)
        prompt_embeds = self.decoder.embed_tokens(batch["prompt_input_ids"])
        caption_embeds = self.decoder.embed_tokens(batch["caption_input_ids"])
        visual_embeds = self.connector(visual_tokens, grid_h, grid_w)
        visual_embeds = visual_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        model_inputs = self.build_multimodal_inputs(
            prompt_embeds=prompt_embeds,
            visual_embeds=visual_embeds,
            caption_embeds=caption_embeds,
            prompt_attention_mask=batch["prompt_attention_mask"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
        )
        outputs = self.decoder(**model_inputs)
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        generation_kwargs: dict | None = None,
    ):
        generation_kwargs = generation_kwargs or {}
        visual_tokens, grid_h, grid_w = self.vision_encoder(pixel_values)
        connector_parameter = next(self.connector.parameters())
        visual_tokens = visual_tokens.to(device=connector_parameter.device, dtype=connector_parameter.dtype)
        visual_embeds = self.connector(visual_tokens, grid_h, grid_w)
        prompt_embeds = self.decoder.embed_tokens(prompt_input_ids)
        visual_embeds = visual_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        visual_attention_mask = torch.ones(
            (visual_embeds.shape[0], visual_embeds.shape[1]),
            dtype=prompt_attention_mask.dtype,
            device=visual_embeds.device,
        )
        inputs_embeds = torch.cat([prompt_embeds, visual_embeds], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask.to(visual_embeds.device), visual_attention_mask],
            dim=1,
        )
        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    def trainable_parameter_counts(self) -> tuple[int, int]:
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        connector_params = sum(p.numel() for p in self.connector.parameters())
        return total_trainable, connector_params
