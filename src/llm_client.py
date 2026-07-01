"""llm_client.py — LLM client abstractions.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from litellm import completion  
from preprocessing import image_to_base64_data_uri
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel




logger = logging.getLogger(__name__)


# Abstract interface

@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class BaseLLMClient(ABC):
    """Contract every LLM backend must fulfil."""

    @abstractmethod
    def complete(self, image_path: str | Path, prompt: str, max_tokens: int) -> LLMResponse:
        """Run a single vision-language completion and return a structured response."""


# LiteLLM (cloud) implementation

class LiteLLMClient(BaseLLMClient):
    """Thin wrapper around litellm.completion for cloud vision models."""

    def __init__(self, model_id: str) -> None:
        self._completion = completion
        self.model_id = model_id

    def complete(self, image_path: str | Path, prompt: str, max_tokens: int) -> LLMResponse:

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64_data_uri(image_path)},
                    },
                ],
            }
        ]
        response = self._completion(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            finish_reason=choice.finish_reason,
        )


# HuggingFace local implementation

class HuggingFaceClient(BaseLLMClient):
    """Runs a local Gemma-3 vision model via transformers."""

    def __init__(self, model_id: str, adapter_path: str | None = None) -> None:
     
        self.model_id = model_id
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype="auto", device_map="auto"
        ).eval()
        self._torch = torch
        if adapter_path:
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            logger.info("LoRA adapter loaded from %s", adapter_path)

    def complete(self, image_path: str | Path, prompt: str, max_tokens: int) -> LLMResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            generation = self._model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        decoded = self._processor.decode(
            generation[0][input_len:], skip_special_tokens=True
        )
        return LLMResponse(
            content=decoded,
            prompt_tokens=input_len,
            completion_tokens=len(generation[0]) - input_len,
            finish_reason="stop",
        )
