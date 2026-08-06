import os
import random
from datasets import load_dataset

# Paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def download_coding_pairs():
    print("📥 Downloading coding pairs...")
    ds = load_dataset("bigcode/tiny_bigcode", split="train", streaming=True)
    ds = ds.shuffle(seed=42).take(1000)
    ds = list(ds)
    return [{"prompt": ex["instruction"], "response": ex["output"]} for ex in ds]

def download_conversations():
    print("📥 Downloading conversations...")
    ds = load_dataset("locuslab/SHARE-chat", split="train", streaming=True)
    ds = ds.shuffle(seed=42).take(1000)
    ds = list(ds)
    conversations = []
    for ex in ds:
        turns = []
        prompt = ex.get("prompt", "")
        response = ex.get("response", "")
        if prompt and response:
            turns.append({"role": "user", "content": prompt})
            turns.append({"role": "assistant", "content": response})
        if turns:
            conversations.append({"messages": turns})
    return conversations

if __name__ == "__main__":
    coding_data = download_coding_pairs()
    conv_data = download_conversations()
    print(f"📊 Coding pairs: {len(coding_data)}")
    print(f"💬 Conversations: {len(conv_data)}")

    # Save raw files
    from datasets import Dataset
    Dataset.from_list(coding_data).save_to_disk(os.path.join(RAW_DIR, "coding_pairs"))
    Dataset.from_list(conv_data).save_to_disk(os.path.join(RAW_DIR, "conversations"))
    print("✅ Data downloaded and saved to data/raw/")
