"""
LoRA fine-tuning of Qwen2-1.5B-Instruct on the MR.AsyncFL paper QA dataset.

Usage:
    python train_lora.py
    python train_lora.py --epochs 3 --batch_size 2 --output_dir ./lora_output
    python train_lora.py --dry_run          # test data loading only

Requirements: pip install -r requirements.txt
Hardware: ~8 GB VRAM with 4-bit quantization (RTX 3080 or better recommended)
"""

import argparse
import json
import os
import random
import pathlib
from pathlib import Path

# trl reads its bundled .jinja templates without declaring UTF-8, which crashes
# on Windows (cp1252 default). Force read_text to default to UTF-8 before
# importing trl. Safe no-op on Linux/macOS or when PYTHONUTF8=1 is set.
_orig_read_text = pathlib.Path.read_text
def _read_text_utf8(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _read_text_utf8

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
DATASET_FILE = "lora_dataset.jsonl"
OUTPUT_DIR = "./lora_output"
MAX_SEQ_LENGTH = 2048

# LoRA hyperparameters
LORA_R = 16           # rank — controls model capacity vs memory
LORA_ALPHA = 32       # scaling factor (usually 2×r)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 2         # per-device batch size; increase if VRAM allows
GRAD_ACCUM_STEPS = 8   # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LR_SCHEDULE = "cosine"
EVAL_RATIO = 0.1       # fraction of data used for validation

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str) -> tuple[Dataset, Dataset]:
    """Load JSONL, format to text, and split into train/val."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)

    n_val = max(1, int(len(records) * EVAL_RATIO))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Dataset loaded: {len(train_records)} train, {len(val_records)} val")
    return Dataset.from_list(train_records), Dataset.from_list(val_records)


def format_chat_to_text(tokenizer, example: dict) -> str:
    """Apply Qwen2 chat template to a messages list and return the full string."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str, use_4bit: bool = True):
    """Load Qwen2-1.5B-Instruct with optional 4-bit quantization and LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right"   # required for SFTTrainer
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    return tokenizer, model


def apply_lora(model):
    """Wrap model with LoRA adapter."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer, model = load_model_and_tokenizer(MODEL_ID, use_4bit=not args.no_quant)
    model = apply_lora(model)

    print(f"Loading dataset from {DATASET_FILE}")
    train_ds, val_ds = load_dataset_from_jsonl(DATASET_FILE)

    if args.dry_run:
        print("Dry run: data loaded successfully. Exiting.")
        sample = format_chat_to_text(tokenizer, train_ds[0])
        print(f"\nSample training text (first 500 chars):\n{sample[:500]}")
        return

    # Format dataset using the chat template
    def preprocess(example):
        return {"text": format_chat_to_text(tokenizer, example)}

    train_ds = train_ds.map(preprocess)
    val_ds = val_ds.map(preprocess)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULE,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=1,           # ~6 optimizer steps/epoch with 90 examples
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",     # change to "wandb" or "tensorboard" for tracking
        max_length=MAX_SEQ_LENGTH,   # renamed from max_seq_length in trl >= 0.13
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


# ─── Inference helper ─────────────────────────────────────────────────────────

def generate(model_dir: str, prompt: str) -> str:
    """Load fine-tuned model and generate an answer."""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()

    messages = [
        {"role": "system", "content": "You are an expert AI assistant specializing in federated learning and the MR.AsyncFL framework."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization (uses more VRAM)")
    parser.add_argument("--dry_run", action="store_true", help="Load data and model, then exit")
    parser.add_argument("--infer", type=str, default=None, help="Run inference with a prompt (requires trained model)")
    args = parser.parse_args()

    if args.infer:
        if not os.path.isdir(args.output_dir):
            print(f"No trained model found at {args.output_dir}. Train first.")
            return
        answer = generate(args.output_dir, args.infer)
        print(f"\nAnswer:\n{answer}")
    else:
        train(args)


if __name__ == "__main__":
    main()
