"""
Data preparation script for the AI Coding Assistant MVP.

Generates a small self-contained hybrid dataset:
- Coding instruction pairs (prompt -> code)
- Multi-turn conversations (chat)

No network access required. Output: data/processed/{train,val}.jsonl
"""
import json
import os
import random

random.seed(42)

DATA_OUT = "data/processed"
os.makedirs(DATA_OUT, exist_ok=True)

CODESYS = (
    "You are a helpful coding assistant. "
    "Respond with clean, correct code and minimal explanation."
)
CHATSYS = (
    "You are a helpful technical assistant. "
    "Help users solve problems concisely."
)

CODING_PAIRS = [
    {
        "prompt": "Write a Python function to reverse a string.",
        "response": (
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
        ),
    },
    {
        "prompt": "Write a Python function to check if a number is prime.",
        "response": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n**0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
    {
        "prompt": "Write a Python function to merge two sorted lists.",
        "response": (
            "def merge_sorted(a, b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            result.append(a[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(b[j])\n"
            "            j += 1\n"
            "    result.extend(a[i:])\n"
            "    result.extend(b[j:])\n"
            "    return result\n"
        ),
    },
    {
        "prompt": "Write a JavaScript function to debounce a callback.",
        "response": (
            "function debounce(fn, delay) {\n"
            "  let timer;\n"
            "  return function(...args) {\n"
            "    clearTimeout(timer);\n"
            "    timer = setTimeout(() => fn.apply(this, args), delay);\n"
            "  };\n"
            "}\n"
        ),
    },
    {
        "prompt": "Write a bash script to find the 5 largest files in a directory.",
        "response": (
            "#!/bin/bash\n"
            "du -ah \"$1\" | sort -rh | head -5\n"
        ),
    },
    {
        "prompt": "Write a C++ function to compute factorial iteratively.",
        "response": (
            "long long factorial(int n) {\n"
            "    long long result = 1;\n"
            "    for (int i = 2; i <= n; ++i)\n"
            "        result *= i;\n"
            "    return result;\n"
            "}\n"
        ),
    },
    {
        "prompt": "Write a Python function to read a JSON file safely.",
        "response": (
            "import json\n"
            "\n"
            "def read_json(path):\n"
            "    with open(path, 'r') as f:\n"
            "        return json.load(f)\n"
        ),
    },
    {
        "prompt": "Write a TypeScript function to clamp a number between min and max.",
        "response": (
            "function clamp(value: number, min: number, max: number): number {\n"
            "  return Math.min(Math.max(value, min), max);\n"
            "}\n"
        ),
    },
    {
        "prompt": "Write a Python function to count words in a string.",
        "response": (
            "def count_words(text):\n"
            "    return len(text.split())\n"
        ),
    },
    {
        "prompt": "Write a bash one-liner to replace 'foo' with 'bar' in all .txt files.",
        "response": (
            "sed -i 's/foo/bar/g' *.txt\n"
        ),
    },
]

CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "What's the difference between a list and a tuple in Python?"},
            {"role": "assistant", "content": (
                "A list is mutable (you can change it after creation), "
                "while a tuple is immutable. Lists use [] and tuples use (). "
                "Tuples are slightly faster and can be used as dictionary keys."
            )},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How do I exit a Vim editor?"},
            {"role": "assistant", "content": (
                "Press Esc, then type :wq and press Enter to save and quit. "
                "To quit without saving, type :q! instead."
            )},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain what garbage collection means."},
            {"role": "assistant", "content": (
                "Garbage collection is automatic memory management. "
                "The runtime detects objects no longer in use and frees their memory, "
                "so you don't have to deallocate manually like in C/C++."
            )},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What does 'git rebase' do?"},
            {"role": "assistant", "content": (
                "Git rebase moves your branch's commits on top of another branch, "
                "creating a linear history. It rewrites commit hashes, "
                "so avoid rebasing branches that others have based work on."
            )},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How can I make my Python code faster?"},
            {"role": "assistant", "content": (
                "Profile first to find bottlenecks. Common wins: use built-in "
                "functions, list comprehensions, avoid global lookups, "
                "and consider NumPy for numerical work or Cython for hot loops."
            )},
        ]
    },
]


def build_dataset():
    formatted = []

    for pair in CODING_PAIRS:
        formatted.append({
            "messages": [
                {"role": "system", "content": CODESYS},
                {"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["response"]},
            ]
        })

    for conv in CONVERSATIONS:
        formatted.append({
            "messages": [
                {"role": "system", "content": CHATSYS},
            ] + conv["messages"]
        })

    random.shuffle(formatted)
    return formatted


def main():
    data = build_dataset()
    split_idx = int(len(data) * 0.8)
    train_set = data[:split_idx]
    val_set = data[split_idx:]

    train_path = os.path.join(DATA_OUT, "train.jsonl")
    val_path = os.path.join(DATA_OUT, "val.jsonl")

    with open(train_path, "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
    with open(val_path, "w") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")

    print(f"Train samples: {len(train_set)}  -> {train_path}")
    print(f"Val samples:   {len(val_set)}  -> {val_path}")


if __name__ == "__main__":
    main()
