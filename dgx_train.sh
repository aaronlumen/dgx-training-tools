#!/usr/bin/env bash
# dgx_train.sh — LLaMA-Factory / Unsloth training menu for NVIDIA DGX Spark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SAVES_DIR="$SCRIPT_DIR/saves"
CONFIG_DIR="$SCRIPT_DIR/configs"
CORPUS_DIR="$SCRIPT_DIR/corpus"
PYTHON="$HOME/jupyterlab/.venv/bin/python3"
[[ ! -x "$PYTHON" ]] && PYTHON="python3"

mkdir -p "$CONFIG_DIR" "$CORPUS_DIR" "$SAVES_DIR"

# ── colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()     { echo -e "${RED}[ERR]${NC}  $*" >&2; exit 1; }
hr()      { echo -e "${CYAN}────────────────────────────────────────────────────${NC}"; }

# ── dependency checks ────────────────────────────────────────────────────────
check_dep() {
    local pkg="$1" import="$2"
    if ! "$PYTHON" -c "import $import" &>/dev/null; then
        warn "$pkg not found — installing..."
        "$PYTHON" -m pip install --quiet "$pkg"
    fi
}

ensure_deps() {
    check_dep pymupdf fitz
    check_dep python-docx docx
}

ensure_unsloth() {
    if ! "$PYTHON" -c "from unsloth import FastLanguageModel" &>/dev/null; then
        info "Installing Unsloth from local wheels…"
        local zoo="$SCRIPT_DIR/unsloth_zoo-2026.4.9-py3-none-any.whl"
        local whl=""
        for name in "unsloth-2026.4.8-py3-none-any.whl" "unsloth-2026.4.7-py3-none-any.whl"; do
            [[ -f "$SCRIPT_DIR/$name" ]] && { whl="$SCRIPT_DIR/$name"; break; }
        done
        [[ -z "$whl" ]]  && die "Unsloth wheel not found in $SCRIPT_DIR"
        [[ ! -f "$zoo" ]] && die "unsloth_zoo wheel not found in $SCRIPT_DIR"
        "$PYTHON" -m pip install --quiet "$zoo" "$whl"
        success "Unsloth installed."
    else
        success "Unsloth already installed."
    fi
}

# ── training runner ──────────────────────────────────────────────────────────
run_trainer() {
    local yaml="$1"
    cd "$SCRIPT_DIR"
    "$PYTHON" "$SCRIPT_DIR/unsloth_trainer.py" "$yaml"
}

# ── PDF → text ────────────────────────────────────────────────────────────────
convert_pdf() {
    ensure_deps
    hr
    echo -e "${BOLD}PDF → Text Conversion${NC}"
    hr
    read -rp "PDF file or directory path: " PDF_INPUT
    [[ -z "$PDF_INPUT" ]] && { warn "No input given."; return; }

    read -rp "Output directory [${CORPUS_DIR}]: " OUT_DIR
    OUT_DIR="${OUT_DIR:-$CORPUS_DIR}"
    mkdir -p "$OUT_DIR"

    "$PYTHON" - "$PDF_INPUT" "$OUT_DIR" <<'PYEOF'
import sys, pathlib, fitz

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
files = list(src.glob("**/*.pdf")) if src.is_dir() else [src]
print(f"Converting {len(files)} PDF(s)...")
for f in files:
    try:
        doc = fitz.open(f)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        out = dst / (f.stem + ".txt")
        out.write_text(text, encoding="utf-8")
        print(f"  ✓ {f.name} → {out.name}  ({len(text):,} chars)")
    except Exception as e:
        print(f"  ✗ {f.name}: {e}")
print("Done.")
PYEOF
    success "PDFs converted → $OUT_DIR"
}

# ── DOCX → text ───────────────────────────────────────────────────────────────
convert_docx() {
    ensure_deps
    hr
    echo -e "${BOLD}DOCX → Text Conversion${NC}"
    hr
    read -rp "DOCX file or directory path: " DOCX_INPUT
    [[ -z "$DOCX_INPUT" ]] && { warn "No input given."; return; }

    read -rp "Output directory [${CORPUS_DIR}]: " OUT_DIR
    OUT_DIR="${OUT_DIR:-$CORPUS_DIR}"
    mkdir -p "$OUT_DIR"

    "$PYTHON" - "$DOCX_INPUT" "$OUT_DIR" <<'PYEOF'
import sys, pathlib, docx

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
files = list(src.glob("**/*.docx")) if src.is_dir() else [src]
print(f"Converting {len(files)} DOCX file(s)...")
for f in files:
    try:
        doc = docx.Document(f)
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        out = dst / (f.stem + ".txt")
        out.write_text(text, encoding="utf-8")
        print(f"  ✓ {f.name} → {out.name}  ({len(text):,} chars)")
    except Exception as e:
        print(f"  ✗ {f.name}: {e}")
print("Done.")
PYEOF
    success "DOCX files converted → $OUT_DIR"
}

# ── Text dir → JSONL ──────────────────────────────────────────────────────────
load_text_dir() {
    hr
    echo -e "${BOLD}Text Directory → Training JSONL${NC}"
    hr
    read -rp "Directory of .txt files [${CORPUS_DIR}]: " TXT_DIR
    TXT_DIR="${TXT_DIR:-$CORPUS_DIR}"
    [[ ! -d "$TXT_DIR" ]] && { warn "Directory not found: $TXT_DIR"; return; }

    read -rp "Output JSONL name [corpus_$(date +%Y%m%d).jsonl]: " JSONL_NAME
    JSONL_NAME="${JSONL_NAME:-corpus_$(date +%Y%m%d).jsonl}"
    JSONL_OUT="$DATA_DIR/$JSONL_NAME"

    read -rp "Chunk size in characters [2000]: " CHUNK_SIZE
    CHUNK_SIZE="${CHUNK_SIZE:-2000}"

    read -rp "Overlap between chunks [200]: " OVERLAP
    OVERLAP="${OVERLAP:-200}"

    "$PYTHON" - "$TXT_DIR" "$JSONL_OUT" "$CHUNK_SIZE" "$OVERLAP" <<'PYEOF'
import sys, pathlib, json, re

txt_dir  = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
chunk_sz = int(sys.argv[3])
overlap  = int(sys.argv[4])

files = sorted(txt_dir.glob("**/*.txt"))
print(f"Found {len(files)} .txt file(s) in {txt_dir}")

def clean(text):
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk(text, size, ov):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - ov
    return chunks

records, total = [], 0
for f in files:
    try:
        raw = f.read_text(encoding="utf-8", errors="replace")
        text = clean(raw)
        for c in chunk(text, chunk_sz, overlap):
            if len(c.strip()) > 50:
                records.append({"text": c.strip()})
                total += 1
    except Exception as e:
        print(f"  ✗ {f.name}: {e}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"  ✓ {total} chunks → {out_path}")
PYEOF

    DATASET_KEY="${JSONL_NAME%.jsonl}"
    _register_dataset "$DATASET_KEY" "$JSONL_NAME" "text" ""
    success "Dataset registered as: $DATASET_KEY"
}

# ── Load existing JSONL ───────────────────────────────────────────────────────
load_jsonl() {
    hr
    echo -e "${BOLD}Load Corpus JSONL${NC}"
    hr

    read -rp "Path to JSONL file: " JSONL_PATH
    [[ -z "$JSONL_PATH" || ! -f "$JSONL_PATH" ]] && { warn "File not found."; return; }

    JSONL_FILE="$(basename "$JSONL_PATH")"
    DEST="$DATA_DIR/$JSONL_FILE"

    if [[ "$JSONL_PATH" != "$DEST" ]]; then
        cp "$JSONL_PATH" "$DEST"
        info "Copied to $DEST"
    fi

    echo ""
    echo "Format options:"
    echo "  1) Alpaca   {instruction, input, output}"
    echo "  2) ShareGPT {conversations: [{from, value}]}"
    echo "  3) Plain    {text}"
    read -rp "Format [3]: " FMT_CHOICE
    FMT_CHOICE="${FMT_CHOICE:-3}"

    DATASET_KEY="${JSONL_FILE%.jsonl}"
    case "$FMT_CHOICE" in
        1) _register_dataset "$DATASET_KEY" "$JSONL_FILE" "alpaca" "" ;;
        2) _register_dataset "$DATASET_KEY" "$JSONL_FILE" "sharegpt" "" ;;
        *) _register_dataset "$DATASET_KEY" "$JSONL_FILE" "text" "" ;;
    esac

    success "Dataset registered as: $DATASET_KEY"
}

# ── Register dataset in dataset_info.json ────────────────────────────────────
_register_dataset() {
    local key="$1" file="$2" fmt="$3" _="$4"
    local info_file="$DATA_DIR/dataset_info.json"

    "$PYTHON" - "$info_file" "$key" "$file" "$fmt" <<'PYEOF'
import sys, json, pathlib

info_file = pathlib.Path(sys.argv[1])
key, file, fmt = sys.argv[2], sys.argv[3], sys.argv[4]

data = json.loads(info_file.read_text()) if info_file.exists() else {}

if fmt == "sharegpt":
    data[key] = {
        "file_name": file,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"}
    }
elif fmt == "alpaca":
    data[key] = {"file_name": file}
else:
    data[key] = {"file_name": file, "formatting": "text", "columns": {"prompt": "text"}}

info_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(f"  Registered '{key}' in dataset_info.json")
PYEOF
}

# ── Training config builder ───────────────────────────────────────────────────
configure_training() {
    hr
    echo -e "${BOLD}Training Configuration${NC}"
    hr

    echo ""
    echo -e "${YELLOW}── Model ──${NC}"
    echo "  Common models:"
    echo "    google/gemma-3-4b-it"
    echo "    google/gemma-3-12b-it"
    echo "    Qwen/Qwen3-4B-Instruct"
    echo "    Qwen/Qwen3-8B-Instruct"
    echo "    meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "    mistralai/Mistral-7B-Instruct-v0.3"
    read -rp "Model name or local path: " MODEL
    [[ -z "$MODEL" ]] && { warn "Model required."; return; }

    echo ""
    echo -e "${YELLOW}── Training Stage ──${NC}"
    echo "  1) SFT  — supervised fine-tuning (default)"
    echo "  2) PT   — continued pre-training"
    read -rp "Stage [1]: " STAGE_CHOICE
    case "${STAGE_CHOICE:-1}" in
        2) STAGE="pt" ;;
        *) STAGE="sft" ;;
    esac

    echo ""
    echo -e "${YELLOW}── Fine-tuning Method ──${NC}"
    echo "  1) LoRA  — adapter layers (default, recommended)"
    echo "  2) Full  — full parameter fine-tuning (needs ~2x VRAM)"
    read -rp "Method [1]: " FT_CHOICE
    case "${FT_CHOICE:-1}" in
        2) FT_TYPE="full" ;;
        *) FT_TYPE="lora" ;;
    esac

    LORA_RANK=8; LORA_ALPHA=8; LORA_DROPOUT=0.05
    if [[ "$FT_TYPE" == "lora" ]]; then
        echo ""
        echo -e "${YELLOW}── LoRA Parameters ──${NC}"
        read -rp "LoRA rank [8]: "    LORA_RANK;    LORA_RANK="${LORA_RANK:-8}"
        read -rp "LoRA alpha [${LORA_RANK}]: " LORA_ALPHA; LORA_ALPHA="${LORA_ALPHA:-$LORA_RANK}"
        read -rp "LoRA dropout [0.05]: " LORA_DROPOUT; LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
    fi

    echo ""
    echo -e "${YELLOW}── Dataset ──${NC}"
    echo "Available datasets:"
    "$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$DATA_DIR/dataset_info.json').read_text())
for k in d: print('  •', k)
" 2>/dev/null || echo "  (none registered yet)"
    read -rp "Dataset name(s), comma-separated: " DATASET
    [[ -z "$DATASET" ]] && { warn "Dataset required."; return; }

    read -rp "Max samples per dataset (blank = all): " MAX_SAMPLES
    read -rp "Context cutoff length [2048]: " CUTOFF_LEN
    CUTOFF_LEN="${CUTOFF_LEN:-2048}"

    echo ""
    echo -e "${YELLOW}── Batch & Training ──${NC}"
    read -rp "Per-device train batch size [2]: "    BATCH_SIZE; BATCH_SIZE="${BATCH_SIZE:-2}"
    read -rp "Gradient accumulation steps [4]: "    GRAD_ACC;   GRAD_ACC="${GRAD_ACC:-4}"
    read -rp "Number of epochs [3]: "               EPOCHS;     EPOCHS="${EPOCHS:-3}"
    read -rp "Learning rate [1e-4]: "               LR;         LR="${LR:-1.0e-4}"
    read -rp "LR scheduler [cosine]: "              LR_SCHED;   LR_SCHED="${LR_SCHED:-cosine}"
    read -rp "Warmup ratio [0.1]: "                 WARMUP;     WARMUP="${WARMUP:-0.1}"

    echo ""
    echo -e "${YELLOW}── Output ──${NC}"
    MODEL_SLUG="$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
    DEFAULT_OUT="saves/${MODEL_SLUG}/${FT_TYPE}/${STAGE}"
    read -rp "Output directory [${DEFAULT_OUT}]: " OUT_DIR
    OUT_DIR="${OUT_DIR:-$DEFAULT_OUT}"
    read -rp "Save steps [500]: "                  SAVE_STEPS; SAVE_STEPS="${SAVE_STEPS:-500}"
    read -rp "Report to [none/wandb/tensorboard]: " REPORT_TO;  REPORT_TO="${REPORT_TO:-none}"

    # ── Build YAML ──────────────────────────────────────────────────────────────
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    YAML_FILE="$CONFIG_DIR/train_${TIMESTAMP}.yaml"

    {
        echo "### model"
        echo "model_name_or_path: $MODEL"
        echo "trust_remote_code: true"
        echo ""
        echo "### method"
        echo "stage: $STAGE"
        echo "do_train: true"
        echo "finetuning_type: $FT_TYPE"
        if [[ "$FT_TYPE" == "lora" ]]; then
            echo "lora_rank: $LORA_RANK"
            echo "lora_alpha: $LORA_ALPHA"
            echo "lora_dropout: $LORA_DROPOUT"
            echo "lora_target: all"
        fi
        echo ""
        echo "### dataset"
        echo "dataset: $DATASET"
        echo "cutoff_len: $CUTOFF_LEN"
        [[ -n "$MAX_SAMPLES" ]] && echo "max_samples: $MAX_SAMPLES"
        echo "dataloader_num_workers: 4"
        echo ""
        echo "### output"
        echo "output_dir: $OUT_DIR"
        echo "logging_steps: 10"
        echo "save_steps: $SAVE_STEPS"
        echo "plot_loss: true"
        echo "overwrite_output_dir: true"
        echo "report_to: $REPORT_TO"
        echo ""
        echo "### train"
        echo "per_device_train_batch_size: $BATCH_SIZE"
        echo "gradient_accumulation_steps: $GRAD_ACC"
        echo "learning_rate: $LR"
        echo "num_train_epochs: $EPOCHS"
        echo "lr_scheduler_type: $LR_SCHED"
        echo "warmup_ratio: $WARMUP"
        echo "bf16: true"
    } > "$YAML_FILE"

    echo ""
    hr
    echo -e "${BOLD}Generated config:${NC} $YAML_FILE"
    hr
    cat "$YAML_FILE"
    hr

    read -rp "Launch training now? [y/N]: " LAUNCH
    if [[ "${LAUNCH,,}" == "y" ]]; then
        info "Starting training…"
        run_trainer "$YAML_FILE"
        success "Training complete. Outputs: $OUT_DIR"
    else
        info "Config saved. Run manually:"
        echo "  $PYTHON $SCRIPT_DIR/unsloth_trainer.py $YAML_FILE"
    fi
}

# ── Resume training ───────────────────────────────────────────────────────────
resume_training() {
    hr
    echo -e "${BOLD}Resume Training from Checkpoint${NC}"
    hr
    echo "Recent configs:"
    ls -t "$CONFIG_DIR"/*.yaml 2>/dev/null | head -10 | nl -w2 -s') ' || { warn "No configs found."; return; }
    read -rp "Config file path (or number above): " CONFIG_INPUT

    if [[ "$CONFIG_INPUT" =~ ^[0-9]+$ ]]; then
        YAML_FILE="$(ls -t "$CONFIG_DIR"/*.yaml 2>/dev/null | sed -n "${CONFIG_INPUT}p")"
    else
        YAML_FILE="$CONFIG_INPUT"
    fi
    [[ ! -f "$YAML_FILE" ]] && { warn "Config not found: $YAML_FILE"; return; }

    info "Running with $YAML_FILE"
    run_trainer "$YAML_FILE"
    success "Training complete."
}

# ── Run existing YAML directly ────────────────────────────────────────────────
run_yaml() {
    hr
    echo -e "${BOLD}Run Existing Config${NC}"
    hr
    echo "Recent configs:"
    ls -t "$CONFIG_DIR"/*.yaml 2>/dev/null | head -10 | nl -w2 -s') ' || { warn "No configs found."; return; }
    read -rp "Config file path (or number above): " CONFIG_INPUT

    if [[ "$CONFIG_INPUT" =~ ^[0-9]+$ ]]; then
        YAML_FILE="$(ls -t "$CONFIG_DIR"/*.yaml 2>/dev/null | sed -n "${CONFIG_INPUT}p")"
    else
        YAML_FILE="$CONFIG_INPUT"
    fi
    [[ ! -f "$YAML_FILE" ]] && { warn "Config not found: $YAML_FILE"; return; }

    hr
    cat "$YAML_FILE"
    hr
    read -rp "Launch this config? [y/N]: " CONFIRM
    [[ "${CONFIRM,,}" != "y" ]] && { info "Cancelled."; return; }

    info "Starting training…"
    run_trainer "$YAML_FILE"
    success "Done."
}

# ── GPU status ────────────────────────────────────────────────────────────────
show_gpu() {
    hr
    echo -e "${BOLD}GPU Status${NC}"
    hr
    nvidia-smi 2>/dev/null || { warn "nvidia-smi not found"; return; }
    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{printf "  GPU:  %s\n  VRAM: %s MiB total | %s MiB used | %s MiB free\n  Temp: %s°C | Util: %s%%\n", $1,$2,$3,$4,$5,$6}'
}

# ── Main menu ─────────────────────────────────────────────────────────────────
main_menu() {
    while true; do
        echo ""
        hr
        echo -e "  ${BOLD}DGX Spark — Unsloth Training Menu${NC}"
        echo -e "  NVIDIA GB10  •  Ubuntu 24.04  •  Unsloth + TRL"
        hr
        echo ""
        echo "  ${YELLOW}Data Preparation${NC}"
        echo "    1) Convert PDF(s) to text"
        echo "    2) Convert DOCX(s) to text"
        echo "    3) Load text files from directory → JSONL"
        echo "    4) Import existing corpus JSONL"
        echo ""
        echo "  ${YELLOW}Training${NC}"
        echo "    5) Configure & launch new training run"
        echo "    6) Run an existing YAML config"
        echo "    7) Resume from checkpoint"
        echo ""
        echo "  ${YELLOW}Utilities${NC}"
        echo "    8) Show GPU status"
        echo "    9) List registered datasets"
        echo "   10) Check / install Unsloth"
        echo "    0) Exit"
        echo ""
        read -rp "  Choice: " CHOICE

        case "$CHOICE" in
            1) convert_pdf ;;
            2) convert_docx ;;
            3) load_text_dir ;;
            4) load_jsonl ;;
            5) configure_training ;;
            6) run_yaml ;;
            7) resume_training ;;
            8) show_gpu ;;
            9)
                hr
                echo -e "${BOLD}Registered Datasets${NC}"
                hr
                "$PYTHON" -c "
import json, pathlib
d = json.loads(pathlib.Path('$DATA_DIR/dataset_info.json').read_text())
for k,v in d.items():
    print(f'  {k:35s}  {v.get(\"file_name\", v.get(\"hf_hub_url\",\"\"))}')
" 2>/dev/null || warn "Could not read dataset_info.json"
                ;;
            10) ensure_unsloth ;;
            0) echo ""; info "Bye."; exit 0 ;;
            *) warn "Invalid choice." ;;
        esac
    done
}

# ── entrypoint ────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
show_gpu
main_menu
