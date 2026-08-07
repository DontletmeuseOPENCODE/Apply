"""
CLI wrapper for the AI Coding Assistant.

Usage:
    python src/cli.py "Write a function to sort a list"
    python src/cli.py -s "You are a Python expert" "Explain decorators"
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import chat


def main():
    parser = argparse.ArgumentParser(
        description="AI Coding Assistant - CPU-optimized LLM"
    )
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the model")
    parser.add_argument(
        "-s", "--system",
        default="You are a helpful coding and technical assistant.",
        help="System prompt",
    )
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else "Hello! Who are you?"
    print(f"\nUser: {prompt}")
    print(f"\nAssistant: {chat(prompt, args.system)}")


if __name__ == "__main__":
    main()
