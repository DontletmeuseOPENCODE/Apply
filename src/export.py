import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-2b-it"
ADAPTER_PATH = "outputs/lora-adapters"
MERGED_PATH = "outputs/merged-model"

print("🔄 Loading base model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, padding_side="right", use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("🔗 Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("🧩 Merging LoRA into base model...")
merged = model.merge_and_unload()

print(f"💾 Saving merged model to {MERGED_PATH}...")
merged.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("✅ Merged model saved.")
