# Arabic Document OCR Fine-Tuning with Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Unsloth](https://img.shields.io/badge/Unsloth-LoRA-green.svg)](https://github.com/unslothai/unsloth)
[![FastAPI](https://img.shields.io/badge/FastAPI-serving-teal.svg)](https://fastapi.tiangolo.com/)

> **A production-grade pipeline for fine-tuning Vision-Language Models (VLMs) on Arabic document understanding using knowledge distillation — from cloud teacher model to efficient deployable local model.**

---

## Project Overview

Arabic government and legal documents are structurally dense and linguistically specific: they mix Hijri and Gregorian dates, contain official seals and stamps, use honorific routing conventions, and must never have Arabic names transliterated or translated. Off-the-shelf OCR tools return flat text and miss all of this structure.

This project solves that with a **knowledge distillation pipeline**:
- A large cloud model (Gemini Flash, the *teacher*) labels document images with rich structured JSON
- That labeled data trains a small local model (Gemma 3 4B, the *student*) via LoRA
- The result is a model that runs on-premise — critical for legal/government documents that cannot leave a client's infrastructure

**What it extracts (10+ structured categories):**

| Category | Fields | Use Case |
|----------|--------|----------|
| Document Classification | Type, subtype, category, languages | Automatic routing & filing |
| Source Information | Issuing authority, department, document numbers | Provenance tracking |
| Physical Properties | Page quality, watermarks, security patterns | Quality control |
| Official Marks | Seals, stamps (color, position, text), QR codes | Authenticity verification |
| Signatures | Signatories, titles, approval chains | Authorization tracking |
| Routing | Addressed to, CC, forwarded to, file references | Workflow automation |
| Content | Full text, tables, lists, charts, legal articles | Content analysis & search |
| Structural Elements | Headers, footers, letterheads, margin notes | Layout understanding |
| Attachments | Referenced documents, attachment counts | Completeness checking |
| Quality Metrics | Confidence scores, uncertain elements, review flags | Quality assurance |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: PDF Preprocessing          src/preprocessing.py   │
│  • PDF → per-page images (pdf2image)                        │
│  • Grayscale conversion  → reduces token cost to VLM        │
│  • Resize to max 600px width (maintain aspect ratio)        │
│  • Contrast ×1.5  → compensates for low-quality scans       │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Knowledge Distillation     src/annotator.py       │
│  • Teacher: Gemini Flash (via OpenRouter / LiteLLM)         │
│  • Each page → structured JSON (10+ categories)             │
│  • Append-mode JSONL → resumable if run interrupted         │
│  • Cost tracked per-image  src/cost_tracker.py              │
│  • Pricing: $0.50/1M input + $3.00/1M output tokens         │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Dataset Preparation        src/dataset_builder.py │
│  • Parse & repair teacher JSON output  src/parsing_data.py  │
│  • Split into 2 specialised tasks:                          │
│      Task 1 → content + structural_elements                 │
│      Task 2 → classification, marks, routing, signatures    │
│  • Deduplicate by image path                                │
│  • Hold out 3 PDFs for validation (fixed seed shuffle)      │
│  • Output: LlamaFactory conversation-format JSON            │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: LoRA Fine-Tuning                                  │
│                                                             │
│  Path A — Unsloth  model/unsloth_finetune.py  (DEFAULT)    │
│  • 4-bit quantised load → fits T4 16GB                      │
│  • Custom Triton kernels → ~60% less VRAM vs HF default     │
│  • rank=16, cutoff=4096, batch=1, grad_accum=4              │
│                                                             │
│  Path B — LlamaFactory  ocr_finetune.yaml                   │
│  • CLI/YAML driven, full precision                          │
│  • Recommended for A100/H100 (40GB+ VRAM)                   │
│                                                             │
│  Both paths:  Student model → Gemma 3 4B Instruct           │
│               Monitoring   → Weights & Biases               │
│               Output       → LoRA adapter checkpoint        │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  Serving: FastAPI REST API           src/api.py             │
│  • Model + adapter loaded ONCE at startup (not per-request) │
│  • POST /extract/content   → Task 1 only                    │
│  • POST /extract/metadata  → Task 2 only                    │
│  • POST /extract           → both tasks                     │
│  • GET  /health            → liveness check                 │
│  • Base model OR fine-tuned adapter via ADAPTER_PATH env    │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Arabic-OCR-VLM-Fine-Tuning/
├── config.py                   # Central config: all paths, model IDs, pricing
├── main.py                     # Pipeline entry point with --stage CLI
├── ocr_finetune.yaml           # LlamaFactory training config (large GPU path)
├── requirements.txt
├── .env.example
│
├── src/
│   ├── preprocessing.py        # PDF → image conversion and enhancement
│   ├── annotator.py            # Teacher-model annotation loop
│   ├── cost_tracker.py         # Running token cost tracker
│   ├── dataset_builder.py      # JSONL → LlamaFactory SFT format
│   ├── parsing_data.py         # JSON repair for model outputs
│   ├── prompts.py              # Extraction prompt + task messages
│   ├── evaluation.py           # 3-way comparison: base / fine-tuned / teacher
│   ├── llm_client.py           # LiteLLMClient + HuggingFaceClient abstractions
│   └── api.py                  # FastAPI serving layer
│
├── model/
│   ├── unsloth_finetune.py     # Unsloth LoRA training (T4-compatible)
│   └── OCR_FineTuning.ipynb    # Original Colab notebook (research/exploration)
│
└── data/
    ├── train.json              # Generated by dataset_builder
    └── val.json
```

---

## Key Design Decisions

### 1. Why Knowledge Distillation Instead of Manual Annotation?

Manual annotation of Arabic legal documents requires domain expertise and is expensive. Instead, Gemini Flash (the teacher) labels the data automatically using a carefully engineered extraction prompt. This gives hundreds of training examples at a fraction of the cost, with consistent structured output. The student model (Gemma 3 4B) then learns to replicate that extraction capability at local inference speed.

### 2. Why Two Specialised Tasks?

A single model asked to simultaneously extract full page text, tables, document classification, seals, approval chains, and routing produces worse results than two models each focused on one half. Task 1 handles content and layout; Task 2 handles metadata and document intelligence. This specialisation improves field-level accuracy and lets API callers pay only for the task they need.

### 3. Why LoRA?

Full fine-tuning of a 4B parameter model is impractical without very large GPU resources. LoRA injects small trainable adapter matrices (rank 16 → ~8M parameters) into the frozen base model. Only those adapters are updated during training, then saved as a small checkpoint (~50-100MB) layered on top of the unchanged base model at inference time.

```
Full fine-tuning : update 4,000,000,000 parameters
LoRA (rank 16)   : update       ~8,000,000 parameters  (0.2%)
```

### 4. Why Two Training Backends?

Hardware constraints are real. The Unsloth path (4-bit quantisation + Triton kernels) makes training possible on a T4 16GB GPU — the free Colab tier. The LlamaFactory YAML path is preserved for environments with larger GPUs where full-precision training is preferred. Same data, same model, different execution paths.

### 5. Config-Driven Architecture

All paths, model IDs, and pricing live in a frozen `AppConfig` dataclass loaded once from environment variables. No module reads `os.getenv()` directly — every module takes `cfg` as an argument. This makes each module independently testable with a fake config and the same codebase deployable across Colab, cloud GPU rentals, and local machines by changing only the `.env` file.

---

## Installation

### Prerequisites

```bash
Python 3.10+
GPU: T4 16GB minimum for training (Colab free tier works)
     GTX 1650 4GB: insufficient for 4B VLM training
API keys: HuggingFace, OpenRouter, Weights & Biases (optional)
```

### System dependency (not pip-installable)

```bash
# Ubuntu / Colab
apt-get install -y poppler-utils

# macOS
brew install poppler
```

### Python dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note:** Do not install torch via requirements.txt on Colab — it ships its own CUDA-matched build. For local installs:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### LlamaFactory (optional — only needed for the large-GPU training path)

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# Data paths
DATA_DIR=/path/to/your/data          # contains downloaded_pdfs/ and pdf_images/

# Models
CLOUD_MODEL_ID=openrouter/google/gemini-3-flash-preview
LOCAL_MODEL_ID=google/gemma-3-4b-it

# API keys
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
WANDB_API_KEY=your_wandb_key          # optional

# Serving (api.py)
ADAPTER_PATH=/path/to/checkpoints/ocr-gemma-unsloth/lora-adapter  # optional
```

If `ADAPTER_PATH` is not set, `api.py` serves the base model — useful for comparing base vs fine-tuned through the same endpoint.

---

## Usage

### Run individual pipeline stages

```bash
# Stage 1: Convert PDFs to preprocessed images (CPU, no API key needed)
python main.py --stage preprocess

# Stage 2: Annotate images with teacher model (needs OPENROUTER_API_KEY)
python main.py --stage annotate

# Stage 3: Build train/val JSON from annotations (CPU, no API key needed)
python main.py --stage build

# Stage 4: Fine-tune with Unsloth LoRA (GPU required, T4 16GB minimum)
python main.py --stage train

# Run stages 1-3 in sequence (no training — safe to run on CPU)
python main.py
```

> **Why separate stages?** Annotation (Stage 2) calls the cloud API and costs money. If annotation already ran, you can re-run `--stage build` to rebuild the dataset without re-annotating. Each stage is independently re-entrant.

### Serve the fine-tuned model

```bash
# Set adapter path if you have a trained checkpoint
export ADAPTER_PATH=./checkpoints/ocr-gemma-unsloth/lora-adapter

uvicorn src.api:app --host 0.0.0.0 --port 8000
```

API is now live at `http://localhost:8000`. Interactive docs at `/docs`.

### API endpoints

```bash
# Liveness check
GET /health

# Extract page text, tables, charts, structural elements (Task 1)
POST /extract/content
  body: multipart/form-data  { file: <image.jpg> }

# Extract classification, seals, signatures, routing (Task 2)
POST /extract/metadata
  body: multipart/form-data  { file: <image.jpg> }

# Run both tasks, returns { "content": {...}, "metadata": {...} }
POST /extract
  body: multipart/form-data  { file: <image.jpg> }
```

### Example curl call

```bash
curl -X POST http://localhost:8000/extract/content \
  -F "file=@/path/to/document_page.jpg"
```

---

## Training Configuration

### Unsloth path (default — T4 16GB)

Configured in `model/unsloth_finetune.py` via `UnslothTrainingConfig`:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `load_in_4bit` | `True` | Required to fit 4B model on T4 16GB |
| `lora_r` | 16 | Balanced capacity vs VRAM (YAML used 96 — only viable on A100) |
| `max_seq_length` | 4096 | T4-safe (YAML used 12000 — would OOM on T4) |
| `per_device_train_batch_size` | 1 | Mandatory for VLMs on 16GB |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 |
| `optim` | `adamw_8bit` | 8-bit Adam saves additional VRAM |

For a smoke-test run (verify pipeline end-to-end before committing to full training):

```python
cfg = UnslothTrainingConfig()
cfg.max_steps = 20   # ~2 minutes on T4
train(cfg)
```

### LlamaFactory path (large GPU)

```bash
python -m llamafactory.cli train \
  ocr_finetune.yaml \
  --dataset_dir /content/LlamaFactory/data \
  --model_name_or_path google/gemma-3-4b-it
```

---

## Evaluation

`src/evaluation.py` runs a three-way comparison on held-out validation documents:

| Model | Description |
|-------|-------------|
| Base | Gemma 3 4B with no fine-tuning |
| Fine-tuned | Gemma 3 4B + LoRA adapter from training |
| Teacher | Gemini Flash (cloud, ground truth reference) |

Metrics are computed at the **field level** against the teacher's structured JSON output — not just surface text similarity. This gives a meaningful signal: did fine-tuning actually close the gap to the teacher on the fields that matter (dates, document numbers, routing)?

---

## Deployment Notes

| Environment | Purpose | Notes |
|-------------|---------|-------|
| Google Colab T4 | Training | Free tier; ephemeral sessions; no uptime guarantee |
| RunPod / Vast.ai A10G | Production training | ~$0.30/hr; stable; recommended for full runs |
| Any persistent server | Serving (api.py) | Needs GPU with 16GB+ VRAM for Gemma 3 4B |
| Modal / Replicate / Baseten | Serverless serving | GPU cold-start tradeoff vs persistent cost |

**Colab is for training only — never production serving.** Sessions disconnect, storage is ephemeral, and there is no uptime SLA. Once a checkpoint is saved to Google Drive, download it and deploy `api.py` on persistent infrastructure.
