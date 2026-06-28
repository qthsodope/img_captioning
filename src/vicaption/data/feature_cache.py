from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from PIL import Image
from tqdm import tqdm


def feature_cache_path(feature_dir: str | Path, image_id: str) -> Path:
    """Return a stable cache path for one image_id."""
    digest = hashlib.sha1(image_id.encode("utf-8")).hexdigest()
    return Path(feature_dir) / f"{digest}.pt"


def load_feature_cache(path: str | Path) -> dict:
    """Load one cache file and validate its minimal schema."""
    import torch

    cache_path = Path(path)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty SigLIP feature cache: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu")

    visual_tokens = payload.get("visual_tokens") if isinstance(payload, dict) else None
    if visual_tokens is None or not hasattr(visual_tokens, "shape"):
        raise ValueError(f"Invalid SigLIP feature cache payload: {cache_path}")
    if len(visual_tokens.shape) != 2:
        raise ValueError(f"Expected cached visual_tokens to be 2D: {cache_path}")
    if int(payload.get("grid_h", 0)) <= 0 or int(payload.get("grid_w", 0)) <= 0:
        raise ValueError(f"Invalid cached grid size: {cache_path}")
    return payload


def is_valid_feature_cache(path: str | Path) -> bool:
    """Return True only when a cache file can be loaded and has the expected payload."""
    try:
        load_feature_cache(path)
    except Exception:
        return False
    return True


def save_feature_cache_atomic(path: str | Path, payload: dict) -> None:
    """Write one feature cache file atomically, leaving no corrupt final file on failure."""
    import torch

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{uuid4().hex}.tmp")

    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        if not is_valid_feature_cache(tmp_path):
            raise RuntimeError(f"Temporary SigLIP feature cache is invalid: {tmp_path}")
        tmp_path.replace(cache_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def collect_unique_image_ids(datasets: Iterable) -> list[str]:
    """Collect image IDs from datasets that expose either image_ids or items."""
    seen: set[str] = set()
    image_ids: list[str] = []
    for dataset in datasets:
        candidates = getattr(dataset, "image_ids", None)
        if candidates is None:
            items = getattr(dataset, "items", [])
            candidates = [item["image_id"] for item in items]
        for image_id in candidates:
            if image_id not in seen:
                seen.add(image_id)
                image_ids.append(image_id)
    return image_ids


def precompute_vision_features(
    image_ids: list[str],
    image_dir: str | Path,
    feature_dir: str | Path,
    processor,
    vision_encoder,
    device,
    batch_size: int = 8,
    overwrite: bool = False,
) -> None:
    """Cache frozen SigLIP patch tokens to disk for faster Connector training."""
    import torch

    image_root = Path(image_dir)
    cache_root = Path(feature_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    pending = [
        image_id
        for image_id in image_ids
        if overwrite or not is_valid_feature_cache(feature_cache_path(cache_root, image_id))
    ]
    if not pending:
        return

    vision_encoder.eval()
    for start in tqdm(range(0, len(pending), batch_size), desc="cache SigLIP features"):
        batch_ids = pending[start : start + batch_size]
        images = [Image.open(image_root / image_id).convert("RGB") for image_id in batch_ids]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            visual_tokens, grid_h, grid_w = vision_encoder(pixel_values)

        visual_tokens = visual_tokens.detach().to("cpu", dtype=torch.float16)
        for index, image_id in enumerate(batch_ids):
            cache_path = feature_cache_path(cache_root, image_id)
            save_feature_cache_atomic(
                cache_path,
                {
                    "image_id": image_id,
                    "visual_tokens": visual_tokens[index].contiguous(),
                    "grid_h": int(grid_h),
                    "grid_w": int(grid_w),
                },
            )
