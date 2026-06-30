"""config.py — Central configuration. All paths, model IDs, and pricing live here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path
    pdf_dir: Path
    image_dir: Path
    sft_output_file: Path
    dataset_dir: Path

    @classmethod
    def from_env(cls) -> "PathConfig":
        data_dir = Path(os.environ["DATA_DIR"])
        return cls(
            data_dir=data_dir,
            pdf_dir=data_dir / "downloaded_pdfs",
            image_dir=data_dir / "pdf_images",
            sft_output_file=data_dir / "ocr-image-sft.jsonl",
            dataset_dir=data_dir / "datasets" / "llamafactory-ocr-finetune-data",
        )


@dataclass(frozen=True)
class ModelConfig:
    cloud_model_id: str
    local_model_id: str
    max_tokens_annotation: int
    max_tokens_eval: int
    price_per_1m_input: float
    price_per_1m_output: float

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            cloud_model_id=os.getenv(
                "CLOUD_MODEL_ID", "openrouter/google/gemini-3-flash-preview"
            ),
            local_model_id=os.getenv("LOCAL_MODEL_ID", "google/gemma-3-4b-it"),
            max_tokens_annotation=int(os.getenv("MAX_TOKENS_ANNOTATION", "8000")),
            max_tokens_eval=int(os.getenv("MAX_TOKENS_EVAL", "4096")),
            price_per_1m_input=float(os.getenv("PRICE_PER_1M_INPUT", "0.5")),
            price_per_1m_output=float(os.getenv("PRICE_PER_1M_OUTPUT", "3.0")),
        )


@dataclass(frozen=True)
class PreprocessingConfig:
    max_image_width: int = 600
    contrast_factor: float = 1.5


@dataclass(frozen=True)
class DatasetConfig:
    val_pdf_files: list[str] = field(
        default_factory=lambda: ["0011.pdf", "0006.pdf", "0001.pdf"]
    )
    random_seed: int = 101
    llamafactory_yaml_path: str = (
        "/content/LlamaFactory/examples/train_lora/OCR_file_tuining.yaml"
    )


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    dataset: DatasetConfig

    @classmethod
    def load(cls) -> "AppConfig":
        return cls(
            paths=PathConfig.from_env(),
            model=ModelConfig.from_env(),
            preprocessing=PreprocessingConfig(),
            dataset=DatasetConfig(),
        )
