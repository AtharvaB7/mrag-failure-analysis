"""
InternVL2.5-8B wrapper.

InternVL's HF checkpoints require `trust_remote_code=True` (the model class
itself, along with its dynamic image-tiling preprocessing, ships as custom
code in the repo rather than being a native `transformers` architecture).
Two things this file works around, both confirmed against a real run:

1. `device_map="auto"` crashes on this model with newer transformers/
   accelerate versions (`AttributeError: 'InternVLChatModel' object has no
   attribute 'all_tied_weights_keys'`) -- accelerate's automatic device-map
   inference calls an attribute this custom-code class never defines. Since
   an 8B model in bf16 (~16GB) fits comfortably on a single A100 (40GB) with
   room to spare, we sidestep this entirely: load with no device_map, then
   manually `.to(device)`. This only supports single-GPU placement, which is
   fine for the intended A100 setup; if you need multi-GPU sharding you'd
   need a different workaround (e.g. pinning an older accelerate version).

2. InternVL's `model.chat()` needs `num_patches_list` (how many image-tiles
   belong to each image) and per-image `Image-N: <image>` markers in the
   prompt whenever more than one image is passed -- omitting this silently
   misattributes tiles to the wrong image rather than raising an error. This
   didn't surface in earlier testing because the no-retrieval setting only
   ever passes one image; it would have broken silently the moment
   sparse/dense/hybrid retrieval settings passed multiple images.

If you hit further compatibility issues, swap in `models/llava_next.py`
instead (see configs/model/llava_next.yaml) -- the rest of the pipeline
(retrieval, evaluation, plotting) is agnostic to which secondary model you use.
"""
from __future__ import annotations

import time

import torch

from models.base import VLMResponse
from models.qwen2vl import format_choices  # reuse the same prompt formatting helper

MC_PROMPT_TEMPLATE = """Answer the following multiple-choice question using the images provided as context (if any).
Respond with only the letter of the correct choice.

Question: {question}
Choices:
{choices_block}

Answer:"""


class InternVLWrapper:
    def __init__(self, hf_id: str, dtype: str = "bfloat16", device_map: str = "auto", num_image_tiles: int = 12):
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if device_map not in ("auto", None):
            raise NotImplementedError(
                f"device_map={device_map!r} not supported for InternVL -- this wrapper only "
                "supports single-GPU placement (see module docstring for why)."
            )

        # Deliberately NOT passing device_map here -- see module docstring
        # point 1. Load on CPU first, then move manually.
        self.model = (
            AutoModel.from_pretrained(hf_id, dtype=torch_dtype, trust_remote_code=True)
            .eval()
            .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, use_fast=False)
        self.num_image_tiles = num_image_tiles

    @torch.no_grad()
    def generate(self, question: str, choices: list[str], images: list, max_new_tokens: int = 32) -> VLMResponse:
        from models.internvl_image_utils import load_image_tensor  # dynamic tiling preprocessing

        base_prompt = MC_PROMPT_TEMPLATE.format(question=question, choices_block=format_choices(choices))

        start = time.perf_counter()
        if images:
            per_image_tensors = [load_image_tensor(img, max_num=self.num_image_tiles) for img in images]
            num_patches_list = [t.size(0) for t in per_image_tensors]
            pixel_values = torch.cat(per_image_tensors, dim=0).to(self.model.device, dtype=self.model.dtype)

            # Required whenever >1 image is passed (see module docstring point
            # 2) -- and harmless/equivalent to the single-image call when
            # there's only one, so applied unconditionally for consistency.
            image_markers = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(images)))
            prompt_text = image_markers + base_prompt

            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt_text,
                dict(max_new_tokens=max_new_tokens, do_sample=False),
                num_patches_list=num_patches_list,
            )
        else:
            response = self.model.chat(
                self.tokenizer, None, base_prompt, dict(max_new_tokens=max_new_tokens, do_sample=False)
            )
        latency = time.perf_counter() - start
        return VLMResponse(text=response.strip(), latency_seconds=latency)
