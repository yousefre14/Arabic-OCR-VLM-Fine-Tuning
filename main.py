"""main.py — Pipeline entry point.

Wires config → preprocessing → annotation → dataset building together.
Each stage is independently importable and testable; this file only
orchestrates them.
"""
from __future__ import annotations

import logging
import os
import sys
import yaml
from pathlib import Path

from config import AppConfig
from preprocessing import convert_all_pdfs
from annotator import OCRAnnotator
from dataset_builder import SFTDatasetBuilder
from llm_client import LiteLLMClient
from prompts import ANNOTATION_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_api_keys() -> None:
    """Push API keys from env into the variables litellm expects."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )


def run_pipeline(cfg: AppConfig) -> None:
    # 1. PDF → images
    logger.info("=== Stage 1: PDF → Images ===")
    pdf_images_map = convert_all_pdfs(
        pdf_dir=cfg.paths.pdf_dir,
        image_dir=cfg.paths.image_dir,
        cfg=cfg.preprocessing,
    )

    # 2. Annotate images with cloud model
    logger.info("=== Stage 2: Cloud Annotation ===")
    client = LiteLLMClient(cfg.model.cloud_model_id)
    annotator = OCRAnnotator(client=client, cfg=cfg)
    annotator.annotate_all(pdf_images_map, prompt=ANNOTATION_PROMPT)

    # 3. Build SFT dataset
    logger.info("=== Stage 3: Build SFT Dataset ===")
    builder = SFTDatasetBuilder(cfg)
    train_ds, val_ds = builder.build()
    builder.save(train_ds, val_ds)

    # 4. Verify LlamaFactory YAML 
    yaml_path = Path(cfg.dataset.llamafactory_yaml_path)
    if yaml_path.exists():
        with yaml_path.open() as f:
            lf_config = yaml.safe_load(f)
        logger.info(
            "LlamaFactory base model: %s",
            lf_config.get("model_name_or_path", "<not set>"),
        )
    else:
        logger.warning("LlamaFactory YAML not found at %s — skipping check.", yaml_path)


if __name__ == "__main__":
    cfg = AppConfig.load()
    setup_api_keys()
    run_pipeline(cfg)
