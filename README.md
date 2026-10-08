# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

It is an open-ended personal learning project (roughly a 5-6 month effort), built piece by piece and entirely handwritten.

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name and total VRAM.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware and the user's training-time budget and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic (memory fit and compute time), not an LLM guess.
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick.
4. **Data ingestion**: a Rust pipeline handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1. A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.
- 🔧 **Sizing** (`sizer_core`, Rust): in progress. The Python-to-Rust hand-off (passing hardware info and time budget in, reading `model_config.json` back) is being designed, and the sizing formulas are still to be written.
- ⏳ Dataset specialization matching: not started.
- ⏳ Rust ingestion pipeline: not started.
- ⏳ Training loop: not started.
- ⏳ Inference with KV-cache: not started.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

To build the sizing program (requires [Rust](https://www.rust-lang.org/tools/install)):

```bash
cd sizer_core
cargo build --release
```

Note: the correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only). If the default `pip install torch` does not match your machine, use the selector on the PyTorch website.

## Requirements

- Python 3.x
- Rust toolchain (for `sizer_core`)
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`amd-smi` / `rocm-smi` / `rocminfo`), depending on hardware. CPU-only systems fall back automatically.

## License

TBD
