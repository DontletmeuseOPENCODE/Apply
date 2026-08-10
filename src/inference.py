"""
CPU inference using HuggingFace Transformers directly (no llama.cpp needed).

Usage:
    python src/inference.py "Your prompt here"
    python src/inference.py  # uses default prompt
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "outputs/Apply-model"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 256


def load_model():
    import os
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    label = "Apply" if os.path.exists(MODEL_PATH) else "Apply (base: TinyLlama, not yet trained)"
    print(f"Loading model: {label}")
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    return tokenizer, model


def format_prompt(prompt, system="You are a helpful coding and technical assistant."):
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{prompt}\n<|end|>\n"
        f"<|assistant|>\n"
    )


def chat(prompt, system="You are a helpful coding and technical assistant."):
    tokenizer, model = load_model()
    text = format_prompt(prompt, system)
    inputs = tokenizer(text, return_tensors="pt")

    print("Generating...", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>")[-1].strip()
    return answer


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Write a Python function to reverse a string."
    print(chat(query))
