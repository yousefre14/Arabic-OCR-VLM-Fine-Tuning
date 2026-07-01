"""model/unsloth_finetune.py — Memory-efficient LoRA fine-tuning via Unsloth.
  This file reads those JSON files, re-formats them for Unsloth's
  SFTTrainer, trains, and saves a LoRA adapter checkpoint.
  That checkpoint is then loaded by HuggingFaceClient in api.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ── Unsloth imports (must come before transformers to patch kernels) ──────────
try:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
except ImportError as exc:
    raise ImportError(
        "Unsloth is not installed. Run:\n"
        "  pip install unsloth\n"
        "or in Colab:\n"
        "  !pip install unsloth"
    ) from exc

from trl import SFTConfig, SFTTrainer


# 1. CONFIGURATION: All tuneable knobs in one place

class UnslothTrainingConfig:
    """Single source of truth for every hyperparameter and path."""

    # Model
    base_model_id: str = "google/gemma-3-4b-it"
    load_in_4bit: bool = True          # 4-bit quantisation — QLORA

    # LoRA adapter settings:
    # r (rank): controls adapter capacity. Higher = more params = better fit but more VRAM.
    lora_r: int = 16
    lora_alpha: int = 16               # Usually set equal to r
    lora_dropout: float = 0.0          # Unsloth recommends 0 for speed
    random_state: int = 42

    # Which layers to fine-tune.
    # Setting all True = full LoRA on both vision encoder and language decoder.
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    train_json: str = "data/train.json"
    val_json: str = "data/val.json"
    output_dir: str = "./checkpoints/ocr-gemma-unsloth"

    # Training loop
    per_device_train_batch_size: int = 1   # 1 is mandatory on T4 for VLMs
    gradient_accumulation_steps: int = 4   # effective batch = 1 * 4 = 4
    warmup_steps: int = 5
    max_steps: int = 300                   # set to -1 to use num_train_epochs instead
    num_train_epochs: int = 3              # only used when max_steps == -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    max_seq_length: int = 4096            

    # Precision — auto-detected: bf16 on Ampere+, fp16 otherwise
    @property
    def bf16(self) -> bool:
        return torch.cuda.is_bf16_supported()

    @property
    def fp16(self) -> bool:
        return not self.bf16


def _load_llamafactory_json(path: str | Path) -> list[dict[str, Any]]:
    """Load LlamaFactory-format JSON written by dataset_builder.py."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one LlamaFactory record into Unsloth's expected format.
"""
    try:
        conversations = record["conversations"]
        image_path = record["images"][0]

        # Strip the leading "<image>" token — that's a LlamaFactory placeholder,
        human_text = conversations[0]["value"].replace("<image>", "", 1).strip()
        assistant_text = conversations[1]["value"]

        image = Image.open(image_path).convert("RGB")

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": human_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]
        }
    except (KeyError, FileNotFoundError, OSError) as exc:
        logger.warning("Skipping record — %s", exc)
        return None


def build_hf_dataset(json_path: str | Path):
    """
    Load a LlamaFactory JSON file and return a HuggingFace Dataset
    ready for Unsloth's SFTTrainer.
    """
    from datasets import Dataset  # imported here to keep module importable without HF

    raw = _load_llamafactory_json(json_path)
    converted = [r for rec in raw if (r := _convert_record(rec)) is not None]
    logger.info(
        "Converted %d / %d records (skipped %d unparseable)",
        len(converted), len(raw), len(raw) - len(converted),
    )
    return Dataset.from_list(converted)


# 3. MODEL LOADING
#    This is where Unsloth does its work.
#    FastVisionModel.from_pretrained wraps transformers' model loader but:
#      - quantises weights to 4-bit (NF4 by default) → fits in 16GB
#      - patches attention/MLP layers with Triton kernels → faster backward pass
#    get_peft_model adds LoRA adapter matrices on top.

def load_model_and_processor(cfg: UnslothTrainingConfig):
    """
    Load Gemma-3-4B in 4-bit and attach LoRA adapters.

    Returns (model, processor) — processor handles tokenisation + image encoding.
    """
    logger.info("Loading base model: %s (4-bit=%s)", cfg.base_model_id, cfg.load_in_4bit)

    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.base_model_id,
        load_in_4bit=cfg.load_in_4bit,
        use_gradient_checkpointing="unsloth",  # Unsloth's gradient checkpointing,
                                                # saves ~30% VRAM vs HF default
    )

    logger.info(
        "Adding LoRA adapters: r=%d, alpha=%d, vision=%s, language=%s",
        cfg.lora_r, cfg.lora_alpha, cfg.finetune_vision_layers, cfg.finetune_language_layers,
    )

    # get_peft_model injects small trainable A/B matrices into the frozen base model.
    # Only these adapter matrices are updated during training
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.finetune_vision_layers,
        finetune_language_layers=cfg.finetune_language_layers,
        finetune_attention_modules=cfg.finetune_attention_modules,
        finetune_mlp_modules=cfg.finetune_mlp_modules,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        random_state=cfg.random_state,
        use_rslora=False,
    )

    # Print trainable parameter count 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return model, processor

# 4. TRAINING
#    SFTTrainer from trl handles the training loop.
#    UnslothVisionDataCollator handles batching images + text correctly.

def train(cfg: UnslothTrainingConfig | None = None) -> None:
    """Full training run: load data → load model → train → save adapter."""

    if cfg is None:
        cfg = UnslothTrainingConfig()

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_dataset = build_hf_dataset(cfg.train_json)
    val_dataset   = build_hf_dataset(cfg.val_json)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg)

    # ── Trainer config ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        # If max_steps > 0 it takes priority over num_train_epochs.
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",    # 8-bit Adam from bitsandbytes
        seed=cfg.random_state,
        report_to="wandb",
        run_name="ocr-gemma-unsloth",
        # SFT-specific: we're passing pre-formatted message dicts, not raw text
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=cfg.max_seq_length,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,                              # processor acts as tokenizer for VLMs
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=UnslothVisionDataCollator(model, processor),  # handles image batching
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer_output = trainer.train()
    logger.info("Training complete. Stats: %s", trainer_output.metrics)

    # ── Save adapter ──────────────────────────────────────────────────────────
    # save_pretrained saves ONLY the LoRA adapter weights (small, ~50-100MB),
    adapter_path = Path(cfg.output_dir) / "lora-adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved to: %s", adapter_path)

# 5. ENTRY POINT

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

