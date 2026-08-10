"""
Interactive chat loop in the terminal (like ChatGPT).

Usage:
    python src/chat.py
    python src/chat.py -s "You are a Python expert"

Commands:  type 'quit' or 'exit' to leave, 'clear' to reset history.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "outputs/Apply-model"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS = 256


def load_model():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    trained = os.path.exists(MODEL_PATH)
    print(f"Loading model: Apply" + ("" if trained else " (base: TinyLlama, not yet trained)") + " ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float32,
        device_map="cpu",
    )
    print("Model ready!\n")
    return tokenizer, model


def build_prompt(history, system):
    prompt = f"<|system|>\n{system}\n<|end|>\n"
    for role, content in history:
        prompt += f"<|{role}|>\n{content}\n<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def generate(tokenizer, model, history, system):
    text = build_prompt(history, system)
    inputs = tokenizer(text, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


def main():
    parser = argparse.ArgumentParser(description="Interactive AI Chat")
    parser.add_argument(
        "-s", "--system",
        default="You are a helpful coding and technical assistant.",
        help="System prompt",
    )
    args = parser.parse_args()

    tokenizer, model = load_model()
    history = []

    print("=" * 50)
    print("  Apply - AI Coding Assistant")
    print("  Commands: 'quit' to exit | 'clear' to reset")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[History cleared]\n")
            continue

        history.append(("user", user_input))
        print("Assistant> ", end="", flush=True)
        reply = generate(tokenizer, model, history, args.system)
        print(reply + "\n")
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()
