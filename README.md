# DGX Training Tools

**Author:** Aaron Surina

*Dedicated to my sons, whom I love very much.*

---

A complete toolkit for fine-tuning large language models on an NVIDIA DGX Spark (GB10 GPU) using [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl). Includes an interactive training menu, a YAML-driven trainer, and a legal corpus downloader for building domain-specific datasets.

---

## Hardware & Environment

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GB10 (DGX Spark) — CUDA capability 12.1 |
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.9.0+cu130 (CUDA 13.0 custom build) |
| Venv | `~/jupyterlab/.venv/` |

> **Note:** The standard PyTorch warning about "max supported capability 12.0" on the GB10 is harmless — CUDA 13.0 supports sm_121 at runtime.

---

## Repository Structure

```
dgx-training-tools/
├── dgx_train.sh          # Interactive training menu (main entrypoint)
├── unsloth_trainer.py    # Unsloth/TRL training engine — reads YAML configs
├── legal_downloader.py   # Download legal corpora from public sources
└── README.md
```

The menu-driven workflow lives in `dgx_train.sh`. It handles data prep, dataset registration, config generation, and training launch. All training is executed by `unsloth_trainer.py` using YAML configs.

Data and configs live in the **LLaMA-Factory** directory alongside this repo (`~/LLaMA-Factory/`):

```
~/LLaMA-Factory/
├── data/
│   ├── dataset_info.json         # Dataset registry — all datasets declared here
│   ├── corpus_20260424.jsonl     # Custom local corpora
│   └── *.jsonl                   # Other registered datasets
├── configs/
│   └── train_YYYYMMDD_HHMMSS.yaml  # Generated training configs
└── saves/
    └── <model>/<method>/<stage>/   # Checkpoints and final adapters
```

---

## Quick Start

### 1. Activate the environment

```bash
source ~/jupyterlab/.venv/bin/activate
```

### 2. Set your HuggingFace token

```bash
export HF_TOKEN=your_hf_token_here
# or add to ~/.bashrc:
echo 'export HF_TOKEN=hf_...' >> ~/.bashrc
```

### 3. Install Unsloth (first time only)

```bash
cd ~/LLaMA-Factory
pip install unsloth_zoo-2026.4.9-py3-none-any.whl unsloth-2026.4.8-py3-none-any.whl
```

### 4. Launch the training menu

```bash
cd ~/LLaMA-Factory
./dgx_train.sh
```

### 5. Or run a config directly

```bash
cd ~/LLaMA-Factory
python3 unsloth_trainer.py configs/train_20260424_004450.yaml
```

---

## Training Menu (`dgx_train.sh`)

```
  DGX Spark — Unsloth Training Menu
  NVIDIA GB10  •  Ubuntu 24.04  •  Unsloth + TRL

  Data Preparation
    1) Convert PDF(s) to text
    2) Convert DOCX(s) to text
    3) Load text files from directory → JSONL
    4) Import existing corpus JSONL

  Training
    5) Configure & launch new training run
    6) Run an existing YAML config
    7) Resume from checkpoint

  Utilities
    8) Show GPU status
    9) List registered datasets
   10) Check / install Unsloth
    0) Exit
```

### Data Preparation (options 1–4)

Options 1 and 2 use **PyMuPDF** (`fitz`) and **python-docx** to extract text from PDFs and Word documents. Option 3 chunks `.txt` files with configurable chunk size and overlap and writes them as `{"text": "..."}` JSONL. All of these automatically register the output in `data/dataset_info.json`.

### Training (options 5–7)

Option 5 walks you through an interactive config builder and generates a YAML file in `configs/`. It then optionally launches training immediately. Option 6 lets you pick any existing YAML config and run it. Option 7 resumes training by pointing to a config that has `resume_from_checkpoint` set.

---

## Training Config (YAML)

All training parameters live in a YAML file. Example:

```yaml
### model
model_name_or_path: google/gemma-3-4b-it
trust_remote_code: false

### method
stage: pt                    # pt = continued pretraining, sft = supervised fine-tuning
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 16
lora_dropout: 0.05
# NOTE: "all" triggers a dtype bug in Unsloth's Gemma3 patch.
# Target specific projections instead:
lora_target: q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj

### dataset
dataset: corpus_20260424,wikipedia_en
max_samples: 50000           # cap per dataset — omit for full dataset
cutoff_len: 2048
dataloader_num_workers: 4

### output
output_dir: saves/google-gemma-3-4b-it/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: none              # or: wandb, tensorboard

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4   # effective batch = 2 × 4 = 8
max_grad_norm: 1.0
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

### Dataset registration

Datasets are declared in `data/dataset_info.json`. Three formats are supported:

**Plain text** (pretraining):
```json
"my_corpus": {
    "file_name": "my_corpus.jsonl",
    "formatting": "text",
    "columns": { "prompt": "text" }
}
```

**Alpaca format** (instruction tuning):
```json
"my_sft_data": {
    "file_name": "my_sft.jsonl"
}
```

**HuggingFace Hub** (downloaded on first use):
```json
"wikipedia_en": {
    "hf_hub_url": "olm/olm-wikipedia-20221220",
    "columns": { "prompt": "text" }
}
```

---

## Legal Corpus Downloader (`legal_downloader.py`)

Downloads training data from free public legal sources and registers them in `dataset_info.json`.

### Sources

| Source | Flag | Records | Notes |
|--------|------|---------|-------|
| CourtListener | `courtlistener` | 4M+ opinions | Free; set `CL_TOKEN` env var for higher rate limits |
| Law StackExchange | `stackexchange` | ~60k Q&A | Full data dump via archive.org |
| Harvard Caselaw | `harvardcap` | 6.7M cases | Requires free API key for full text |
| CRS Reports | `crs` | ~15k memos | Congressional Research Service — public domain |
| DOJ OLC | `olc` | ~1.5k memos | Office of Legal Counsel opinions — gold-standard legal reasoning |

### Usage

```bash
cd ~/LLaMA-Factory

# Download 50k CourtListener opinions and register the dataset
python3 ~/dgx-training-tools/legal_downloader.py \
    --source courtlistener \
    --output data/courtlistener.jsonl \
    --max 50000 \
    --register

# Download Law StackExchange Q&A
python3 ~/dgx-training-tools/legal_downloader.py \
    --source stackexchange \
    --output data/law_se.jsonl \
    --register

# Download Harvard CAP (requires free API key from case.law)
export HARVARD_CAP_KEY=your_key_here
python3 ~/dgx-training-tools/legal_downloader.py \
    --source harvardcap \
    --output data/harvard_cap.jsonl \
    --max 100000 \
    --register

# Download CRS policy memos
python3 ~/dgx-training-tools/legal_downloader.py \
    --source crs \
    --output data/crs_reports.jsonl \
    --register

# Download DOJ OLC legal memoranda (requires pymupdf: pip install pymupdf)
python3 ~/dgx-training-tools/legal_downloader.py \
    --source olc \
    --output data/olc_memos.jsonl \
    --register

# Download all sources into data/ directory
python3 ~/dgx-training-tools/legal_downloader.py \
    --source all \
    --output data/ \
    --max 50000 \
    --register
```

### After downloading

The `--register` flag adds each dataset to `data/dataset_info.json`. Then include it in a training config:

```yaml
dataset: corpus_20260424,courtlistener,law_se,crs_reports
max_samples: 50000
```

### API keys and rate limits

| Service | Key needed | Where to get it |
|---------|-----------|-----------------|
| CourtListener | Optional | https://www.courtlistener.com/sign-in/ → Profile → API |
| Harvard CAP | Required for full text | https://case.law/user/register/ |
| HuggingFace | For gated models | https://huggingface.co/settings/tokens |

Set tokens in `~/.bashrc`:
```bash
export HF_TOKEN=hf_...
export CL_TOKEN=your_courtlistener_token
export HARVARD_CAP_KEY=your_harvard_cap_key
```

---

## Tips for the DGX Spark

**`max_samples` is critical.** The full `wikipedia_en` dataset has millions of articles. Without `max_samples`, a 3-epoch run would take months. A value of `50000` per dataset gives a multi-hour run.

**Gradient norm spikes.** Occasional large `grad_norm` values (10k–100k) in the logs are clipped by `max_grad_norm: 1.0` and are harmless — they come from noisy batches in raw text corpora.

**LoRA target for Gemma models.** Unsloth's Gemma3 patch has a dtype bug when `lora_target: all` is used (fp16 cast into bf16 `o_proj`). Use explicit targets instead:
```yaml
lora_target: q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj
```

**Saving the adapter vs. the merged model.** `unsloth_trainer.py` saves the LoRA adapter only. To merge and export a full model:
```python
model.save_pretrained_merged("saves/merged", tokenizer, save_method="merged_16bit")
```

**GGUF export** (for use with llama.cpp):
```python
model.save_pretrained_gguf("saves/gguf", tokenizer, quantization_method="q4_k_m")
```

---

## Dependencies

All required wheels are bundled in `~/LLaMA-Factory/`:

```
torch-2.10.0           (or 2.9.0+cu130 from jupyterlab venv)
transformers-5.5.0
peft-0.19.1
trl-0.19.1
accelerate-1.13.0
datasets-4.3.0
unsloth-2026.4.8
unsloth_zoo-2026.4.9
bitsandbytes-0.49.2
```

Optional for legal downloader:
```bash
pip install pymupdf requests
```

---

## License

Scripts in this repository are released under the MIT License.
Training data downloaded via `legal_downloader.py` is sourced from public domain or openly licensed sources. Verify the license of any dataset before use in commercial applications.
