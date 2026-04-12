# Setup Guide - Arabic OCR Fine-Tuning

## 🚀 Quick Setup (Google Colab - Recommended)

### Step 1: Open in Colab

1. Upload `ocr_finetuning.py` to your Google Drive
2. Open Google Colab: https://colab.research.google.com/
3. File → Open notebook → Upload → Select your file

### Step 2: Set Up API Keys

In Colab, go to the "Secrets" panel (🔑 icon on left sidebar):

```
Key: my-hf
Value: your_huggingface_token

Key: openrouter
Value: your_openrouter_api_key

Key: wandb
Value: your_wandb_api_key (optional)
```

Get your API keys:
- **Hugging Face**: https://huggingface.co/settings/tokens
- **OpenRouter**: https://openrouter.ai/keys
- **Weights & Biases**: https://wandb.ai/authorize

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/gdrive')
```

Create your data directory:
```bash
mkdir -p "/gdrive/MyDrive/VLM Finetuning OCR/assets/downloaded_images/downloaded_pdfs"
```

### Step 4: Upload Your PDFs

1. Go to Google Drive: https://drive.google.com/
2. Navigate to: `VLM Finetuning OCR/assets/downloaded_images/downloaded_pdfs/`
3. Upload your Arabic PDF documents

### Step 5: Run the Notebook

Execute cells in order:
1. Install dependencies (takes ~5 minutes)
2. Login to Weights & Biases
3. Clone and install LlamaFactory
4. Login to Hugging Face
5. Process PDFs → Generate training data → Fine-tune model

---

## 💻 Local Setup (Linux/macOS)

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 50GB+ free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/arabic-ocr-finetuning.git
cd arabic-ocr-finetuning
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases/

### Step 4: Install Python Packages

```bash
pip install -r requirements.txt
```

### Step 5: Install LlamaFactory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LlamaFactory
pip install -e .
cd ..
```

### Step 6: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 7: Prepare Data Directory

```bash
mkdir -p data/downloaded_pdfs
mkdir -p data/pdf_images
mkdir -p data/datasets
```

---

## 🔧 Configuration

### API Keys (.env file)

```env
# Hugging Face
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter (for Gemini API)
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxx

# Weights & Biases (optional)
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=arabic-ocr-finetuning
WANDB_ENTITY=your-username
```

### Data Paths

Edit the data directory in the script:

```python
# For local setup
data_dir = "./data"

# For Google Colab
data_dir = "/gdrive/MyDrive/VLM Finetuning OCR/assets/downloaded_images"
```

### LlamaFactory Config (OCR_file_tuning.yaml)

```yaml
# Model Configuration
model_name_or_path: google/gemma-3-4b-it
stage: sft
do_train: true
finetuning_type: lora

# LoRA Configuration
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

# Dataset
dataset_dir: ./data
dataset: ocr_finetune
template: gemma
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 4

# Training Hyperparameters
output_dir: ./checkpoints/ocr-gemma-lora
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
lr_scheduler_type: cosine
warmup_steps: 100
fp16: true

# Logging
logging_steps: 10
save_steps: 500
eval_steps: 500
evaluation_strategy: steps
save_total_limit: 3

# Weights & Biases
report_to: wandb
run_name: ocr-gemma-lora-v1
```

---

## 📁 Dataset Preparation

### Dataset Format (dataset_info.json)

Create `LlamaFactory/data/dataset_info.json`:

```json
{
  "ocr_finetune": {
    "file_name": "/path/to/data/datasets/llamafactory-ocr-finetune-data/train-v1.json",
    "file_sha1": "auto",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  },
  "ocr_finetune_val": {
    "file_name": "/path/to/data/datasets/llamafactory-ocr-finetune-data/val-v1.json",
    "file_sha1": "auto",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

---

## 🎯 Usage Workflow

### Phase 1: PDF Preprocessing (5-10 minutes for 100 pages)

```python
from pdf2image import convert_from_path
from preprocessing import preprocessing_images

# Convert PDFs to images
for pdf_file in pdf_files:
    images = convert_from_path(pdf_file, dpi=150)
    for i, img in enumerate(images):
        processed = preprocessing_images(img, max_width=600)
        processed.save(f"output/page_{i:03d}.jpg")
```

### Phase 2: Knowledge Distillation (Cost: ~$5-10 for 1000 pages)

```python
import litellm
from tqdm import tqdm

# Process each image with Gemini
for img_path in tqdm(image_paths):
    response = completion(
        model="openrouter/google/gemini-3-flash-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": extraction_prompt},
                {"type": "image_url", "image_url": {"url": image_to_base64(img_path)}}
            ]
        }],
        max_tokens=8000
    )
    
    save_response(response, output_file)
```

### Phase 3: Dataset Formatting (1-2 minutes)

```python
# Split into train/val
train_ds = []
val_ds = []
val_pdf_files = ['0011.pdf', '0006.pdf', '0001.pdf']

for record in jsonl_data:
    sft_record = format_for_llamafactory(record)
    
    if record['pdf_name'] in val_pdf_files:
        val_ds.append(sft_record)
    else:
        train_ds.append(sft_record)

# Save formatted datasets
save_json(train_ds, "train-v1.json")
save_json(val_ds, "val-v1.json")
```

### Phase 4: Fine-Tuning (2-4 hours on V100)

```bash
# Start training
python -m llamafactory.cli train \
  OCR_file_tuning.yaml \
  --dataset_dir ./data \
  --model_name_or_path google/gemma-3-4b-it
```

Monitor progress:
- **Weights & Biases**: https://wandb.ai/your-username/arabic-ocr-finetuning
- **Local logs**: `checkpoints/ocr-gemma-lora/trainer_log.jsonl`

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```yaml
# In OCR_file_tuning.yaml, reduce batch size:
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### Issue: Poppler Not Found (pdf2image error)

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install -y poppler-utils

# macOS
brew install poppler

# Windows
# Download from: https://github.com/oschwartz10612/poppler-windows/releases/
# Add bin/ directory to PATH
```

### Issue: OpenRouter API Rate Limit

**Solution:**
```python
# Add delay between requests
import time

for img in images:
    response = completion(...)
    time.sleep(1)  # 1 second delay
```

### Issue: JSON Parsing Errors

**Solution:**
```python
from json_repair import loads

# Use robust JSON parsing
try:
    data = loads(response_text)
except:
    print(f"Failed to parse: {response_text[:100]}")
    continue
```

---

## 📊 Cost Estimation

### Knowledge Distillation Phase

**Gemini 3 Flash Pricing:**
- Input: $0.50 per 1M tokens
- Output: $3.00 per 1M tokens

**Estimated tokens per page:**
- Input: ~1,500 tokens (prompt + image)
- Output: ~3,000 tokens (structured JSON)

**Cost for 1000 pages:**
```
Input:  (1000 × 1500 / 1,000,000) × $0.50 = $0.75
Output: (1000 × 3000 / 1,000,000) × $3.00 = $9.00
Total: $9.75
```

### Fine-Tuning Phase

**GPU Costs (Google Colab Pro+):**
- V100 GPU: $50/month (unlimited)
- Training time: 2-4 hours
- Effective cost: ~$5

### Total Project Cost

- Development (1000 pages): ~$15
- Production inference: $1.50 per 1000 pages (local GPU)

---

## ✅ Verification

After setup, verify everything works:

```bash
# Test imports
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import pdf2image; print('pdf2image OK')"

# Test LlamaFactory
python -m llamafactory.cli --help

# Test Hugging Face login
python -c "from huggingface_hub import whoami; print(whoami())"
```

---

## 🎓 Next Steps

1. **Process your first PDF**: Start with 1-2 sample documents
2. **Review outputs**: Check the generated JSON for quality
3. **Adjust prompts**: Refine extraction prompts if needed
4. **Scale up**: Process your full dataset
5. **Fine-tune model**: Train on the distilled data
6. **Evaluate**: Test on validation set
7. **Deploy**: Export model for production use

---

**Need help?** Open an issue on GitHub or check the documentation!
