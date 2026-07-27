"""
LLaVA-NeXT (llava-v1.6-mistral-7b) wrapper -- fallback secondary model if
InternVL2.5-8B is difficult to reproduce (custom-code compatibility issues,
etc.). Fully supported natively in `transformers` (no trust_remote_code
needed), which is exactly why it's a good fallback: fewer moving parts.
"""
from __future__ import annotations

import time

import torch

from models.base import VLMResponse
from models.qwen2vl import format_choices

MC_PROMPT_TEMPLATE = """Answer the following multiple-choice question using the images provided as context (if any).
Respond with only the letter of the correct choice.

Question: {question}
Choices:
{choices_block}"""


class LlavaNextWrapper:
    def __init__(self, hf_id: str, dtype: str = "bfloat16", device_map: str = "auto"):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        torch_dtype = getattr(torch, dtype)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_id, dtype=torch_dtype, device_map=device_map
        ).eval()
        self.processor = LlavaNextProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def generate(self, question: str, choices: list[str], images: list, max_new_tokens: int = 32) -> VLMResponse:
        prompt_text = MC_PROMPT_TEMPLATE.format(question=question, choices_block=format_choices(choices))
        image_token_block = "<image>" * len(images) if images else ""
        full_prompt = f"[INST] {image_token_block}\n{prompt_text} [/INST]"

        inputs = self.processor(
            text=full_prompt, images=images if images else None, return_tensors="pt"
        ).to(self.model.device)

        start = time.perf_counter()
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        latency = time.perf_counter() - start

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return VLMResponse(text=output_text.strip(), latency_seconds=latency)
