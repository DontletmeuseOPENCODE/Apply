"""
Merge LoRA adapters into the base model and save a full CPU-ready model.

No GGUF conversion here (that requires llama.cpp build tools).
This produces a HuggingFace-format merged model usable on CPU.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "outputs/lora-adapters"
MERGED_PATH = "outputs/merged-model"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapters into base model...")
merged = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
merged.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("Done!")
