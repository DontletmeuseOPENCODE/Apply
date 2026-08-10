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
import time
import traceback

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
    model.generation_config.max_length = None
    print("Model ready!\n")
    return tokenizer, model


def build_prompt(history, system):
    messages = [{"role": "system", "content": system}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    return messages


def generate(tokenizer, model, history, system):
    messages = build_prompt(history, system)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return answer, input_len, len(new_tokens)


def ask_dev_mode():
    print("=" * 50)
    print("  Apply - AI Coding Assistant")
    print("=" * 50)
    choice = input("Enable Dev Mode? Shows errors, token counts, timing. (y/n): ").strip().lower()
    dev = choice in ("y", "yes", "tak", "1")
    if dev:
        print("[Dev Mode: ON]")
    else:
        print("[Dev Mode: OFF]")
    print("  Commands: 'quit' to exit | 'clear' to reset | 'dev' to toggle\n")
    return dev


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
    dev_mode = ask_dev_mode()

    if dev_mode:
        import psutil
        mem = psutil.virtual_memory()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[DEV] Model params: {param_count / 1e6:.1f}M")
        print(f"[DEV] RAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB")
        print(f"[DEV] CPU threads: {os.cpu_count()}")
        print()

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
        if user_input.lower() == "dev":
            dev_mode = not dev_mode
            print(f"[Dev Mode: {'ON' if dev_mode else 'OFF'}]\n")
            continue

        history.append(("user", user_input))
        print("Assistant> ", end="", flush=True)

        try:
            t0 = time.time()
            reply, in_tokens, out_tokens = generate(tokenizer, model, history, args.system)
            elapsed = time.time() - t0
            print(reply + "\n")

            if dev_mode:
                print(f"  [DEV] input tokens: {in_tokens} | output tokens: {out_tokens}")
                print(f"  [DEV] time: {elapsed:.2f}s | tokens/s: {out_tokens / elapsed:.1f}")
                mem = psutil.virtual_memory()
                print(f"  [DEV] RAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB ({mem.percent}%)")
                print()

            history.append(("assistant", reply))

        except Exception as e:
            print(f"[ERROR] {e}\n")
            if dev_mode:
                traceback.print_exc()
                print()


if __name__ == "__main__":
    main()
