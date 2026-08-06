import json
import os
import random
from datasets import load_from_disk

DATA_RAW = "data/raw"
DATA_OUT = "data/processed"
os.makedirs(DATA_OUT, exist_ok=True)
random.seed(42)

# System instructions
CODESYS = "You are a helpful coding assistant. Respond only with code and minimal explanation."
CHATSYS = "You are a helpful technical assistant. Help users solve problems."

def format_coding_pair(pair):
    messages = [
        {"role": "system", "content": CODESYS},
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["response"]}
    ]
    return {"messages": messages}

def process_all(split_ratio=0.9):
    print("📂 Loading raw datasets...")
    coding = load_from_disk(os.path.join(DATA_RAW, "coding_pairs"))
    convs = load_from_disk(os.path.join(DATA_RAW, "conversations"))

    formatted = []
    for pair in coding:
        formatted.append(format_coding_pair(pair[0]))

    for conv in convs:
        messages = [{"role": "system", "content": CHATSYS}] + conv["messages"]
        formatted.append({"messages": messages})

    random.shuffle(formatted)

    split_idx = int(len(formatted) * split_ratio)
    train_set = formatted[:split_idx]
    val_set = formatted[split_idx:]

    with open(os.path.join(DATA_OUT, "train.jsonl"), "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(DATA_OUT, "val.jsonl"), "w") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")

    print(f"📊 Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"✅ Saved to {DATA_OUT}/")

if __name__ == "__main__":
    process_all()
