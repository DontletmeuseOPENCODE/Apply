import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_NAME = "google/gemma-2b-it"
DATA_PATH = "data/processed/train.jsonl"
VAL_PATH = "data/processed/val.jsonl"
OUTPUT_DIR = "outputs/lora-adapters"
LOG_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print("🚀 Setting up tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print("✅ LoRA model ready.")

    print("📚 Loading data...")
    train_ds = load_dataset("json", data_files=DATA_PATH)["train"]
    val_ds = load_dataset("json", data_files=VAL_PATH)["train"]

    def tokenize_fn(example):
        messages = example["messages"]
        prompt = ""
        for msg in messages:
            prompt += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"
        inputs = tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    train_ds = train_ds.map(tokenize_fn, remove_columns=["messages"])
    val_ds = val_ds.map(tokenize_fn, remove_columns=["messages"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        logging_dir=LOG_DIR,
        report_to=[],
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        evaluation_strategy="steps",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    print("🏁 Starting training...")
    trainer.train()

    print("💾 Saving adapter weights...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done!")

if __name__ == "__main__":
    main()
