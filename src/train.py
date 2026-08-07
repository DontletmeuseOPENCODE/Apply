"""
CPU-only LoRA fine-tuning for the AI Coding Assistant.

No GPU/CUDA required. Uses full-precision LoRA (no 4-bit quantization
during training, since bitsandbytes needs CUDA).

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (open, no gating).
"""
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs/lora-adapters"
LOG_DIR = "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MAX_LEN = 512


def format_messages(messages):
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n<|end|>\n".join(parts) + "\n<|end|>\n"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="right", use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (CPU, float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading and tokenizing data...")
    train_ds = load_dataset("json", data_files=DATA_PATH)["train"]
    val_ds = load_dataset("json", data_files=VAL_PATH)["train"]

    def tokenize_fn(example):
        text = format_messages(example["messages"])
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    train_ds = train_ds.map(tokenize_fn, remove_columns=["messages"])
    val_ds = val_ds.map(tokenize_fn, remove_columns=["messages"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=5,
        logging_dir=LOG_DIR,
        report_to=[],
        learning_rate=2e-4,
        save_total_limit=1,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter weights...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Adapters saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
