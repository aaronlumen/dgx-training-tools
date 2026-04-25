#!/usr/bin/env python3
"""Unsloth LoRA/SFT trainer — reads YAML configs produced by dgx_train.sh"""
import sys
import os
import json
import pathlib
import subprocess

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR   = SCRIPT_DIR / "data"


def _ensure_unsloth():
    try:
        from unsloth import FastLanguageModel  # noqa
        return
    except ImportError:
        pass
    zoo = SCRIPT_DIR / "unsloth_zoo-2026.4.9-py3-none-any.whl"
    for name in ("unsloth-2026.4.8-py3-none-any.whl", "unsloth-2026.4.7-py3-none-any.whl"):
        whl = SCRIPT_DIR / name
        if whl.exists():
            print(f"[INFO] Installing Unsloth from {whl.name} …")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   str(zoo), str(whl)])
            return
    raise RuntimeError(
        "Unsloth wheel not found in script directory.\n"
        "Expected: unsloth-2026.4.8-py3-none-any.whl + unsloth_zoo-2026.4.9-py3-none-any.whl"
    )


def _load_one_dataset(name, max_samples=None):
    from datasets import load_dataset

    info_path = DATA_DIR / "dataset_info.json"
    info = json.loads(info_path.read_text())
    if name not in info:
        raise KeyError(f"'{name}' not found in dataset_info.json")

    entry   = info[name]
    cols    = entry.get("columns", {})
    src_col = cols.get("prompt", cols.get("text", "text"))

    if "file_name" in entry:
        fp = DATA_DIR / entry["file_name"]
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")
        ds = load_dataset("json", data_files=str(fp), split="train")
    elif "hf_hub_url" in entry:
        print(f"  Downloading from HuggingFace: {entry['hf_hub_url']} …")
        ds = load_dataset(
            entry["hf_hub_url"],
            name=entry.get("subset"),
            split=entry.get("split", "train"),
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Cannot load '{name}': no file_name or hf_hub_url")

    # normalise to a single 'text' column
    if src_col != "text":
        if src_col in ds.column_names:
            ds = ds.rename_column(src_col, "text")
        elif "text" not in ds.column_names:
            fallback = ds.column_names[0]
            print(f"  [WARN] column '{src_col}' not found, using '{fallback}'")
            ds = ds.rename_column(fallback, "text")

    extras = [c for c in ds.column_names if c != "text"]
    if extras:
        ds = ds.remove_columns(extras)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    return ds


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 unsloth_trainer.py <config.yaml>")
        sys.exit(1)

    try:
        import yaml
    except ImportError:
        print("[ERR] pyyaml not found: pip install pyyaml")
        sys.exit(1)

    cfg_path = pathlib.Path(sys.argv[1])
    if not cfg_path.is_absolute():
        cfg_path = SCRIPT_DIR / cfg_path
    if not cfg_path.exists():
        print(f"[ERR] Config not found: {cfg_path}")
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"[INFO] Config: {cfg_path.name}")

    _ensure_unsloth()

    from unsloth import FastLanguageModel
    from datasets import concatenate_datasets
    from trl import SFTTrainer, SFTConfig

    # ── hypers ───────────────────────────────────────────────────────────────────
    model_name   = cfg["model_name_or_path"]
    cutoff_len   = int(cfg.get("cutoff_len", 2048))
    lora_rank    = int(cfg.get("lora_rank", 8))
    lora_alpha   = int(cfg.get("lora_alpha", lora_rank))
    lora_dropout = float(cfg.get("lora_dropout", 0.05))
    max_samples  = int(cfg["max_samples"]) if cfg.get("max_samples") else None
    use_bf16     = bool(cfg.get("bf16", True))

    output_dir = cfg.get("output_dir", "saves/output")
    if not os.path.isabs(output_dir):
        output_dir = str(SCRIPT_DIR / output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    lora_target    = cfg.get("lora_target", "all")
    target_modules = "all-linear" if lora_target in ("all", "all-linear") else lora_target.split(",")

    # ── model ────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cutoff_len,
        dtype=None,
        load_in_4bit=False,
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── datasets ─────────────────────────────────────────────────────────────────
    names = [n.strip() for n in str(cfg["dataset"]).split(",")]
    print(f"\n[INFO] Loading datasets: {names}")
    loaded = []
    for name in names:
        try:
            ds = _load_one_dataset(name, max_samples)
            print(f"  ✓ {name}: {len(ds):,} samples")
            loaded.append(ds)
        except Exception as exc:
            print(f"  ✗ {name}: {exc} — skipping")

    if not loaded:
        print("[ERR] No datasets loaded. Aborting.")
        sys.exit(1)

    train_ds = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
    print(f"[INFO] Total training samples: {len(train_ds):,}")

    # ── training ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cutoff_len,
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            num_train_epochs=int(cfg.get("num_train_epochs", 3)),
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=int(cfg.get("logging_steps", 10)),
            optim="adamw_8bit",
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            save_steps=int(cfg.get("save_steps", 500)),
            output_dir=output_dir,
            report_to=cfg.get("report_to", "none"),
            overwrite_output_dir=cfg.get("overwrite_output_dir", True),
            dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
        ),
    )

    print("\n[INFO] Training …")
    result = trainer.train()

    # ── save ─────────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Saving model → {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if cfg.get("plot_loss", False):
        try:
            import matplotlib.pyplot as plt
            logs   = [x for x in trainer.state.log_history if "loss" in x]
            steps  = [x["step"] for x in logs]
            losses = [x["loss"] for x in logs]
            plt.figure(figsize=(10, 4))
            plt.plot(steps, losses)
            plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
            plot_path = pathlib.Path(output_dir) / "training_loss.png"
            plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            print(f"[INFO] Loss plot → {plot_path}")
        except Exception as e:
            print(f"[WARN] Could not save loss plot: {e}")

    print(f"\n[OK]  Training complete.")
    print(f"      Metrics:    {result.metrics}")
    print(f"      Model at:   {output_dir}")


if __name__ == "__main__":
    main()
