"""preprocessing.py — Image preprocessing utilities for OCR pipeline."""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm

from config import PreprocessingConfig

logger = logging.getLogger(__name__)


# Image processing

def preprocess_image(image: Image.Image, cfg: PreprocessingConfig) -> Image.Image:
    """Convert a PIL image to a greyscale, contrast-enhanced, resized form.

    Steps:
      1. Greyscale  — reduces size and simplifies OCR input.
      2. Resize     — caps width at max_image_width, preserving aspect ratio.
      3. Contrast   — sharpens faint text with a configurable factor.
    """
    grey = image.convert("L")

    if grey.width > cfg.max_image_width:
        ratio = cfg.max_image_width / grey.width
        new_height = int(grey.height * ratio)
        grey = grey.resize((cfg.max_image_width, new_height), Image.LANCZOS)

    enhanced = ImageEnhance.Contrast(grey).enhance(cfg.contrast_factor)
    return enhanced


def image_to_base64_data_uri(image_path: str | Path) -> str:
    """Encode an image file as a base64 data URI suitable for multimodal APIs."""
    image_path = Path(image_path)
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.suffix.lstrip(".").lower()
    mime_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    return f"data:{mime_type};base64,{encoded}"


# PDF → image conversion

def convert_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    cfg: PreprocessingConfig,
) -> list[Path]:
    """Convert every page of a PDF to a preprocessed JPEG.
    Returns a list of saved image paths.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir) / pdf_path.name
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(str(pdf_path))
    saved_paths: list[Path] = []

    for i, page in enumerate(pages):
        processed = preprocess_image(page, cfg)
        dest = output_dir / f"page_{i + 1:03d}.jpg"
        processed.save(str(dest), "JPEG")
        saved_paths.append(dest)

    logger.info("Converted %d pages from '%s'", len(saved_paths), pdf_path.name)
    return saved_paths


def convert_all_pdfs(
    pdf_dir: Path,
    image_dir: Path,
    cfg: PreprocessingConfig,
) -> dict[str, list[Path]]:
    """Convert every PDF in pdf_dir and return a mapping of pdf_name → image paths."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    result: dict[str, list[Path]] = {}

    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        paths = convert_pdf_to_images(pdf_file, image_dir, cfg)
        result[pdf_file.name] = paths

    return result
