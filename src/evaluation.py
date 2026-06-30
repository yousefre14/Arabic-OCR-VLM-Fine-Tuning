"""evaluation.py — Compare local base model vs cloud model on a sample image.

"""
from __future__ import annotations

import logging
from pathlib import Path

from config import AppConfig
from llm_client import BaseLLMClient, HuggingFaceClient, LiteLLMClient
from prompts import prompt

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Runs a side-by-side comparison of two LLM backends on a sample image."""

    def __init__(
        self,
        local_client: BaseLLMClient,
        cloud_client: BaseLLMClient,
        cfg: AppConfig,
    ) -> None:
        self._local = local_client
        self._cloud = cloud_client
        self._cfg = cfg

    def evaluate(self, sample_image_path: str | Path, prompt: str = prompt) -> dict[str, str]:
        """Run both models on the sample and return their text outputs."""
        sample_image_path = Path(sample_image_path)
        results: dict[str, str] = {}

        logger.info("Running LOCAL model on %s", sample_image_path.name)
        local_resp = self._local.complete(
            image_path=sample_image_path,
            prompt=prompt,
            max_tokens=self._cfg.model.max_tokens_eval,
        )
        results["local"] = local_resp.content
        logger.info("Local finish_reason: %s", local_resp.finish_reason)

        logger.info("Running CLOUD model on %s", sample_image_path.name)
        cloud_resp = self._cloud.complete(
            image_path=sample_image_path,
            prompt=prompt,
            max_tokens=self._cfg.model.max_tokens_eval,
        )
        results["cloud"] = cloud_resp.content
        logger.info("Cloud finish_reason: %s", cloud_resp.finish_reason)

        return results

    def print_comparison(self, results: dict[str, str]) -> None:
        separator = "=" * 60
        for label, text in results.items():
            print(f"\n{separator}")
            print(f"  {label.upper()} MODEL OUTPUT")
            print(separator)
            print(text)


# Convenience factory so callers don't need to know concrete types

def build_evaluator(cfg: AppConfig) -> ModelEvaluator:
    local_client = HuggingFaceClient(cfg.model.local_model_id)
    cloud_client = LiteLLMClient(cfg.model.cloud_model_id)
    return ModelEvaluator(local_client, cloud_client, cfg)
