"""
Qwen2-VL-7B-Instruct wrapper.

Requires: transformers>=4.46 (Qwen2-VL support), qwen-vl-utils for the
`process_vision_info` helper (image resizing to the model's expected patch
grid). Both are pip-installable; only the actual weight download needs
huggingface.co access.
"""
from __future__ import annotations

import time

import torch

from models.base import VLMResponse

MC_PROMPT_TEMPLATE = """Answer the following multiple-choice question using the images provided as context (if any).
Respond with only the letter of the correct choice.

Question: {question}
Choices:
{choices_block}

Answer:"""


def format_choices(choices: list[str]) -> str:
    letters = "ABCDEFGH"
    return "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))


class Qwen2VLWrapper:
    def __init__(self, hf_id: str, dtype: str = "bfloat16", device_map: str = "auto",
                 min_pixels: int = 200704, max_pixels: int = 1003520):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch_dtype, device_map=device_map
        ).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id, min_pixels=min_pixels, max_pixels=max_pixels)

    @torch.no_grad()
    def generate(self, question: str, choices: list[str], images: list, max_new_tokens: int = 32) -> VLMResponse:
        from qwen_vl_utils import process_vision_info

        prompt_text = MC_PROMPT_TEMPLATE.format(question=question, choices_block=format_choices(choices))
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        start = time.perf_counter()
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        latency = time.perf_counter() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return VLMResponse(text=output_text.strip(), latency_seconds=latency)
