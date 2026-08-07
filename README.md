# Apply

Fine-tuning a lightweight LLM with LoRA for coding + general chat.
Designed to run entirely on **CPU** — no GPU required.

## Structure
- `src/data_download.py` — generates a self-contained hybrid dataset (coding + chat)
- `src/train.py` — CPU-only LoRA fine-tuning on TinyLlama 1.1B
- `src/export.py` — merges LoRA adapters into the base model
- `src/inference.py` — CPU inference via HuggingFace Transformers
- `src/cli.py` — CLI wrapper

## Quick Start
```bash
# 1. Generate dataset (instant, no network)
python src/data_download.py

# 2. Fine-tune (CPU)
python src/train.py

# 3. Merge adapters
python src/export.py

# 4. Run inference
python src/inference.py "Write a Python function to reverse a string"

# 5. Or use the CLI
python src/cli.py "Explain how git rebase works"
```

## Tech Stack
| Component    | Choice                          |
|--------------|---------------------------------|
| Base Model   | TinyLlama 1.1B Chat             |
| Fine-tuning  | PEFT (LoRA) + Transformers      |
| Runtime      | PyTorch CPU (float32)           |
| Python       | 3.10+                           |

## Hardware Target
- **CPU-only** (Intel Core i7-6600U, 16 GB RAM)
- No CUDA / GPU required
