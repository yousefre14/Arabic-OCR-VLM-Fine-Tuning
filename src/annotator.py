"""annotator.py — Cloud annotation pass: image → LLM → JSONL record.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from config import AppConfig
from cost_tracker import CostTracker
from llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OCRAnnotator:
    """Iterates over PDF image paths, annotates each via an LLM, appends to JSONL.
    """

    def __init__(self, client: BaseLLMClient, cfg: AppConfig) -> None:
        self._client = client
        self._cfg = cfg
        self._cost_tracker = CostTracker(
            price_per_1m_input=cfg.model.price_per_1m_input,
            price_per_1m_output=cfg.model.price_per_1m_output,
        )

    # Public API

    def annotate_all(
        self,
        pdf_images_map: dict[str, list[Path]],
        prompt: str,
    ) -> None:
        """Annotate every image in pdf_images_map and append results to JSONL.

        Args:
            pdf_images_map: mapping of pdf_name → list of image paths.
            prompt: the annotation prompt sent to the LLM.
        """
        output_file = self._cfg.paths.sft_output_file
        already_done = self._load_completed_paths(output_file)
        ix = len(already_done)

        for pdf_name, images in pdf_images_map.items():
            for img_path in tqdm(images, desc=f"Annotating {pdf_name}"):
                img_str = str(img_path)
                if img_str in already_done:
                    logger.debug("Skipping already annotated: %s", img_str)
                    continue

                ix += 1
                self._annotate_single(ix, pdf_name, img_path, prompt, output_file)
                self._cost_tracker.log(ix)

        logger.info(
            "Annotation complete. Total images: %d | Final cost: $%.4f",
            ix,
            self._cost_tracker.total_cost,
        )

    # Private helpers

    def _annotate_single(
        self,
        ix: int,
        pdf_name: str,
        img_path: Path,
        prompt: str,
        output_file: Path,
    ) -> None:
        try:
            response = self._client.complete(
                image_path=img_path,
                prompt=prompt,
                max_tokens=self._cfg.model.max_tokens_annotation,
            )
        except Exception as exc: 
            logger.error("LLM call failed for %s: %s", img_path, exc)
            return

        if response.finish_reason != "stop":
            logger.warning(
                "Unexpected finish_reason '%s' for image %s (ix=%d)",
                response.finish_reason,
                img_path,
                ix,
            )
            return

        record = {
            "id": ix,
            "pdf_name": pdf_name,
            "image_path": str(img_path),
            "model_id": self._cfg.model.cloud_model_id,
            "output": response.content,
        }
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

        self._cost_tracker.update(response.prompt_tokens, response.completion_tokens)

    @staticmethod
    def _load_completed_paths(output_file: Path) -> set[str]:
        """Return the set of image_path values already written to the JSONL."""
        if not output_file.exists():
            return set()
        done: set[str] = set()
        with output_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    done.add(record["image_path"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return done
