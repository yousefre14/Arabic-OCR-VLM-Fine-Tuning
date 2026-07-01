"""main.py — Pipeline entry point.

Wires config → preprocessing → annotation → dataset building → training together.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

from config import AppConfig
from src.annotator import OCRAnnotator
from src.preprocessing import collect_existing_images
from llm_client import LiteLLMClient
from src.prompts import prompt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# GUARDS
def _require_openrouter_key() -> None:
    """Raise early if the cloud API key is missing — before any API call is made."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file.\n"
            "This key is required for the annotation stage (cloud teacher model)."
        )

# STAGES
def stage_preprocess(cfg: AppConfig) -> None:
    """Stage 1: Convert PDFs to preprocessed images."""
    from src.preprocessing import convert_all_pdfs

    logger.info("=== Stage 1: PDF → Images ===")
    pdf_images_map = convert_all_pdfs(
        pdf_dir=cfg.paths.pdf_dir,
        image_dir=cfg.paths.image_dir,
        cfg=cfg.preprocessing,
    )
    total_images = sum(len(v) for v in pdf_images_map.values())
    logger.info("Preprocessing complete: %d PDFs → %d images", len(pdf_images_map), total_images)


def stage_annotate(cfg: AppConfig) -> None:
    """Stage 2: Run teacher model (Gemini) over all images, write JSONL."""

    _require_openrouter_key()

    logger.info("=== Stage 2: Cloud Annotation (Teacher: %s) ===", cfg.model.cloud_model_id)

    # collect_existing_images reads what's already on disk — avoids re-running
    pdf_images_map = collect_existing_images(cfg.paths.image_dir)

    client = LiteLLMClient(cfg.model.cloud_model_id)
    annotator = OCRAnnotator(client=client, cfg=cfg)
    annotator.annotate_all(pdf_images_map, prompt=prompt)
    logger.info("Annotation complete. Output: %s", cfg.paths.sft_output_file)


def stage_build(cfg: AppConfig) -> None:
    """Stage 3: Build LlamaFactory-format train/val JSON from annotated JSONL."""
    from src.dataset_builder import SFTDatasetBuilder

    logger.info("=== Stage 3: Build SFT Dataset ===")
    builder = SFTDatasetBuilder(cfg)
    train_ds, val_ds = builder.build()
    builder.save(train_ds, val_ds)

    yaml_path = Path(cfg.dataset.llamafactory_yaml_path)
    if yaml_path.exists():
        with yaml_path.open() as f:
            lf_config = yaml.safe_load(f)
        logger.info(
            "LlamaFactory YAML check — base model: %s",
            lf_config.get("model_name_or_path", "<not set>"),
        )
    else:
        logger.info(
            "LlamaFactory YAML not found at %s — skipping check "
            "(not required unless using LlamaFactory training path).",
            yaml_path,
        )


def stage_train(cfg: AppConfig) -> None:
    """
    Stage 4: Fine-tune student model with Unsloth LoRA.
    """
    logger.info("=== Stage 4: LoRA Fine-Tuning (Unsloth) ===")

    try:
        from model.unsloth_finetune import UnslothTrainingConfig, train
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. Run: pip install unsloth\n"
        ) from exc

    train_cfg = UnslothTrainingConfig()
    train_cfg.train_json = str(cfg.paths.dataset_dir / "train-v1.json")
    train_cfg.val_json   = str(cfg.paths.dataset_dir / "val-v1.json")

    logger.info("Training data:  %s", train_cfg.train_json)
    logger.info("Validation data: %s", train_cfg.val_json)
    logger.info("Output dir:      %s", train_cfg.output_dir)

    train(train_cfg)
    logger.info("Training complete. Adapter saved to: %s/lora-adapter", train_cfg.output_dir)

# ORCHESTRATION

_STAGES = {
    "preprocess": stage_preprocess,
    "annotate":   stage_annotate,
    "build":      stage_build,
    "train":      stage_train,
}


def run_pipeline(cfg: AppConfig, stage: str) -> None:
    if stage == "all":
        stage_preprocess(cfg)
        stage_annotate(cfg)
        stage_build(cfg)
        logger.info(
            "Pipeline complete. Run 'python main.py --stage train' "
        )
    elif stage in _STAGES:
        _STAGES[stage](cfg)
    else:
        raise ValueError(f"Unknown stage: '{stage}'. Choose from: {list(_STAGES)} + ['all']")

# ENTRY POINT

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arabic OCR VLM fine-tuning pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  preprocess   Convert PDFs → images (CPU, no API key needed)
  annotate     Label images with teacher model (needs OPENROUTER_API_KEY)
  build        Build train/val JSON from JSONL (CPU, no API key needed)
  train        Fine-tune with Unsloth LoRA (GPU required)
  all          preprocess + annotate + build (default, no training)
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "annotate", "build", "train", "all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = AppConfig.load()
    run_pipeline(cfg, stage=args.stage)