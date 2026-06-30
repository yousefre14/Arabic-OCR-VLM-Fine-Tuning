"""dataset_builder.py — Builds LlamaFactory-format SFT datasets from raw JSONL.
  - Read the annotated JSONL.
  - Split each record into two task conversations (content/structure vs metadata).
  - Separate train / val by PDF name.
  - Shuffle deterministically and write JSON files.

"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from config import AppConfig
from parsing_data import parse_json
from prompts import task_1_message,task_2_message

logger = logging.getLogger(__name__)

SFTRecord = dict[str, Any]


class SFTDatasetBuilder:
    """Transforms annotated JSONL into train/val SFT datasets."""

    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg

    # Public API

    def build(self) -> tuple[list[SFTRecord], list[SFTRecord]]:
        """Read annotations, split, shuffle, and return (train_ds, val_ds)."""
        train_ds: list[SFTRecord] = []
        val_ds: list[SFTRecord] = []
        seen_paths: set[str] = set()

        sft_file = self._cfg.paths.sft_output_file
        if not sft_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {sft_file}")

        with sft_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._process_record(record, train_ds, val_ds, seen_paths)

        rng = random.Random(self._cfg.dataset.random_seed)
        rng.shuffle(train_ds)
        rng.shuffle(val_ds)

        logger.info(
            "Dataset built: %d train records, %d val records",
            len(train_ds),
            len(val_ds),
        )
        return train_ds, val_ds

    def save(self, train_ds: list[SFTRecord], val_ds: list[SFTRecord]) -> None:
        """Persist train and val splits to the configured dataset directory."""
        out_dir = self._cfg.paths.dataset_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(out_dir / "train-v1.json", train_ds)
        self._write_json(out_dir / "val-v1.json", val_ds)
        logger.info("Saved datasets to %s", out_dir)

    # Private helpers

    def _process_record(
        self,
        record: dict[str, Any],
        train_ds: list[SFTRecord],
        val_ds: list[SFTRecord],
        seen_paths: set[str],
    ) -> None:
        image_path = record.get("image_path", "")
        if image_path in seen_paths:
            return
        seen_paths.add(image_path)

        llm_output = parse_json(record.get("output", ""))
        if not llm_output:
            logger.debug("Skipping unparseable record id=%s", record.get("id"))
            return

        task1 = self._make_task1_record(llm_output, image_path)
        task2 = self._make_task2_record(llm_output, image_path)

        target = (
            val_ds
            if record.get("pdf_name") in self._cfg.dataset.val_pdf_files
            else train_ds
        )
        target.append(task1)
        target.append(task2)

    @staticmethod
    def _make_task1_record(llm_output: dict[str, Any], image_path: str) -> SFTRecord:
        output = {
            "content": llm_output.get("content", ""),
            "structural_elements": llm_output.get("structural_elements", ""),
        }
        return _build_sft_record(task_1_message, output, image_path)

    @staticmethod
    def _make_task2_record(llm_output: dict[str, Any], image_path: str) -> SFTRecord:
        # Strip task-1 keys; everything remaining is task-2 metadata
        output = {
            k: v
            for k, v in llm_output.items()
            if k not in {"content", "structural_elements"}
        }
        return _build_sft_record(task_2_message, output, image_path)

    @staticmethod
    def _write_json(path: Path, data: list[SFTRecord]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        logger.info("Wrote %d records to %s", len(data), path)


# Module-level factory helper

def _build_sft_record(
    task_prompt: str,
    output: dict[str, Any],
    image_path: str,
) -> SFTRecord:
    return {
        "conversations": [
            {"from": "human", "value": f"<image>{task_prompt}"},
            {
                "from": "gpt",
                "value": json.dumps(output, ensure_ascii=False, default=str),
            },
        ],
        "images": [image_path],
    }
