"""Build the right VLM wrapper from a Hydra/OmegaConf `model` config node."""
from __future__ import annotations

from typing import Any


def build_model(model_cfg: Any):
    if model_cfg.wrapper == "qwen2vl":
        from models.qwen2vl import Qwen2VLWrapper

        return Qwen2VLWrapper(
            hf_id=model_cfg.hf_id,
            dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            min_pixels=model_cfg.min_pixels,
            max_pixels=model_cfg.max_pixels,
        )

    if model_cfg.wrapper == "internvl":
        from models.internvl import InternVLWrapper

        return InternVLWrapper(
            hf_id=model_cfg.hf_id,
            dtype=model_cfg.dtype,
            device_map=model_cfg.device_map,
            num_image_tiles=model_cfg.num_image_tiles,
        )

    if model_cfg.wrapper == "llava_next":
        from models.llava_next import LlavaNextWrapper

        return LlavaNextWrapper(hf_id=model_cfg.hf_id, dtype=model_cfg.dtype, device_map=model_cfg.device_map)

    raise ValueError(f"Unknown model wrapper: {model_cfg.wrapper}")
