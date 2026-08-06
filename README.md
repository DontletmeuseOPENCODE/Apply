# AI Coding Assistant & Chatbot

Fine-tuning a lightweight LLM (1B-3B params) with LoRA/QLoRA for coding + general chat.
Designed to run on CPU with GGUF export.

## Structure
- `data/`     : datasets and preparation scripts
- `src/`      : source code (fine-tuning, export, inference)
- `logs/`     : training logs
- `outputs/`  : final models and artifacts

## Phases
- Phase 1: Project setup (done)
- Phase 2: Data preparation
- Phase 3: Fine-tuning (LoRA/QLoRA)
- Phase 4: GGUF export + CPU inference
- Phase 5: CLI tool or API wrapper
