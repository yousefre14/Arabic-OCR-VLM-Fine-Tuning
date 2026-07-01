"""api.py — FastAPI serving layer for the fine-tuned OCR model.
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import sys
import io
from fastapi import UploadFile as FU
from starlette.datastructures import UploadFile as StarletteUpload
sys.path.insert(0, str(Path(__file__).parent.parent)) 

from config import AppConfig
from llm_client import HuggingFaceClient
from parsing_data import parse_json
from prompts import task_1_message, task_2_message, prompt

logger = logging.getLogger(__name__)

# 1. APPLICATION STATE

class _AppState:
    client: HuggingFaceClient | None = None
    cfg: AppConfig | None = None


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler: runs setup before the server accepts requests,
    and teardown when it shuts down.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("Loading config...")
    _state.cfg = AppConfig.load()

    model_id = _state.cfg.model.local_model_id
    adapter_path = os.environ.get("ADAPTER_PATH")   # optional LoRA adapter

    logger.info("Loading model: %s", model_id)
    if adapter_path:
        logger.info("Applying LoRA adapter from: %s", adapter_path)
    else:
        logger.warning(
            "ADAPTER_PATH not set — running base model without LoRA adapter. "
        )

    # HuggingFaceClient needs a small extension to accept adapter_path —
    _state.client = HuggingFaceClient(model_id, adapter_path=adapter_path)
    logger.info("Model loaded and ready.")

    yield  # server is live here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing model from memory.")
    _state.client = None


# 2. PYDANTIC RESPONSE MODELS
#    Pydantic models document API contract and give callers
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_id: str
    adapter_loaded: bool


class ExtractionResponse(BaseModel):
    """
    Wraps the raw structured JSON from the model with metadata.
    """
    task: str = Field(..., description="Which task was run: 'content', 'metadata', or 'full'")
    model_id: str
    adapter_loaded: bool
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    data: dict[str, Any] = Field(..., description="Structured extraction output")
    raw_output: str = Field(..., description="Raw model text before JSON parsing")


class ErrorResponse(BaseModel):
    detail: str

# 3. FASTAPI APP

app = FastAPI(
    title="Arabic OCR VLM API",
    description=(
        "Structured data extraction from Arabic document images "
        "using a LoRA fine-tuned Gemma-3-4B vision model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# 4. SHARED HELPER

def _run_extraction(
    image_file: UploadFile,
    prompt: str,
    task_label: str,
    max_tokens: int,
) -> ExtractionResponse:
    """
    Save uploaded image to a temp file, run the model, parse the output.

    Why a temp file?
    HuggingFaceClient.complete() takes a file path (it needs to open the image
    with PIL). UploadFile gives us a file-like object, not a path.
    We write it to a NamedTemporaryFile and pass that path.
    The temp file is deleted automatically when the context manager exits.
    """
    if _state.client is None or _state.cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server startup logs.",
        )

    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/tiff"}
    if image_file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {image_file.content_type}. Use JPEG or PNG.",
        )

    # Write to temp file, run inference
    suffix = Path(image_file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(image_file.file.read())
        tmp.flush()

        try:
            response = _state.client.complete(
                image_path=tmp.name,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.exception("Model inference failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {exc}",
            ) from exc

    # Parse model output — teacher model always returns valid JSON,
    # fine-tuned model should too, but json_repair handles edge cases.
    parsed = parse_json(response.content)
    if parsed is None:
        logger.warning("JSON parsing failed for task=%s. Returning raw output.", task_label)
        parsed = {"error": "json_parse_failed", "raw": response.content}

    return ExtractionResponse(
        task=task_label,
        model_id=_state.cfg.model.local_model_id,
        adapter_loaded=os.environ.get("ADAPTER_PATH") is not None,
        finish_reason=response.finish_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        data=parsed,
        raw_output=response.content,
    )

# 5. ENDPOINTS

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["System"],
)
def health() -> HealthResponse:
    """
    Returns 200 if the model is loaded and the server is ready to accept requests.
    Returns 503 if the model failed to load at startup.
    """
    if _state.client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialised.",
        )
    return HealthResponse(
        status="ok",
        model_id=_state.cfg.model.local_model_id,
        adapter_loaded=os.environ.get("ADAPTER_PATH") is not None,
    )


@app.post(
    "/extract/content",
    response_model=ExtractionResponse,
    summary="Extract page content and structural elements (Task 1)",
    tags=["Extraction"],
)
async def extract_content(
    file: UploadFile = File(..., description="Document image (JPEG or PNG)"),
) -> ExtractionResponse:
    """
    **Task 1** — Extracts:
    - `content`: full text, tables, lists, charts, legal articles, financial data
    - `structural_elements`: headers, footers, letterheads, margin notes

    Use this when you need the actual page text for indexing or search.
    Faster than /extract/metadata because the output schema is simpler.
    """
    return _run_extraction(
        image_file=file,
        prompt=task_1_message,
        task_label="content",
        max_tokens=_state.cfg.model.max_tokens_eval,
    )


@app.post(
    "/extract/metadata",
    response_model=ExtractionResponse,
    summary="Extract document metadata (Task 2)",
    tags=["Extraction"],
)
async def extract_metadata(
    file: UploadFile = File(..., description="Document image (JPEG or PNG)"),
) -> ExtractionResponse:
    """
    **Task 2** — Extracts:
    - `document_classification`, `source`, `physical_properties`
    - `official_marks` (seals, stamps, QR codes)
    - `signatures_authorization`, `routing_distribution`
    - `attachments_references`, `condition_notes`, `confidence_quality`

    Use this for document routing, authenticity checks, and workflow automation.
    """
    return _run_extraction(
        image_file=file,
        prompt=task_2_message,
        task_label="metadata",
        max_tokens=_state.cfg.model.max_tokens_eval,
    )


@app.post(
    "/extract",
    response_model=dict[str, ExtractionResponse],
    summary="Full extraction — runs Task 1 + Task 2 and merges output",
    tags=["Extraction"],
)
async def extract_full(
    file: UploadFile = File(..., description="Document image (JPEG or PNG)"),
) -> dict[str, ExtractionResponse]:
    """
    Runs both Task 1 and Task 2 sequentially on the same image.
    Returns `{"content": ..., "metadata": ...}` with both results.

    Note: This makes two model inference calls. For high-throughput workloads,
    call /extract/content and /extract/metadata independently in parallel.
    """
    # Read the file once, then rewind for the second call
    file_bytes = await file.read()

    def _make_upload(data: bytes, filename: str, content_type: str) -> UploadFile:
        return UploadFile(
            filename=filename,
            file=io.BytesIO(data),
            headers={"content-type": content_type},
        )

    filename = file.filename or "image.jpg"
    content_type = file.content_type or "image/jpeg"

    content_result  = _run_extraction(
        image_file=_make_upload(file_bytes, filename, content_type),
        prompt=task_1_message,
        task_label="content",
        max_tokens=_state.cfg.model.max_tokens_eval,
    )
    metadata_result = _run_extraction(
        image_file=_make_upload(file_bytes, filename, content_type),
        prompt=task_2_message,
        task_label="metadata",
        max_tokens=_state.cfg.model.max_tokens_eval,
    )

    return {"content": content_result, "metadata": metadata_result}
