# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random initialization, no fine-tuning of existing pretrained models) in PyTorch, sized to fit what the user's own hardware can realistically handle. The app detects the user's system specs and available training time, uses an LLM to estimate a feasible parameter count (up to 100M), matches the user's chosen specialization to a curated dataset, and trains the model using a Rust-based data ingestion pipeline feeding into a PyTorch training loop.

## How it works (planned pipeline)

1. **Hardware detection** — detect GPU (NVIDIA/AMD) or CPU-only, and available VRAM.
2. **Sizing** — an LLM wrapper takes the detected hardware info + the user's time budget and estimates an optimal, trainable parameter count (capped at 100M).
3. **Specialization matching** — the user describes what they want the model specialized in; an LLM matches this to a pre-curated dataset (user can override the pick).
4. **Data ingestion** — a Rust pipeline handles tokenizing, batching, and streaming the training data.
5. **Training** — a from-scratch PyTorch model is trained via standard backpropagation.
6. **Output** — a small, locally-runnable, hyperspecialized text model.

## Current status

- ✅ Hardware detection (`architecture.py`) — detects NVIDIA (via `nvidia-smi`), AMD legacy (via `rocm-smi`/`rocminfo`), AMD modern (via `amd-smi`), or falls back to CPU-only.
- ⏳ Sizing/LLM wrapper — in progress.
- ⏳ Dataset specialization matching — not started.
- ⏳ Rust ingestion pipeline — not started.
- ⏳ Training loop — not started.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`rocm-smi`/`rocminfo`/`amd-smi`), depending on hardware. CPU-only systems work fine too — the app falls back automatically.

## License

TBD
