# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

It is an open-ended personal learning project (roughly a 5-6 month effort), built piece by piece and entirely handwritten.

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name and total VRAM.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware and the user's training-time budget and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic (a memory limit and a compute-time limit, taking the smaller), not an LLM guess.
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick.
4. **Data ingestion**: a Rust pipeline handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1. A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. If several GPUs are found, the user picks one. Also asks for the training-time budget in hours. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.
- 🔧 **Sizing** (`sizer_core`, Rust): in progress.
  - Python sends the training hours and the VRAM (MiB) to the Rust program over standard input, one value per line. The Rust side currently reads and echoes them back.
  - Next: parse the inputs, write the memory-limit and time-limit functions, and return the result to Python over standard output.
  - Planned: measure the machine's real speed (FLOPs per second) with a matrix-multiplication benchmark and pass it in as a third value.
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

The Python code currently uses only the standard library, so `requirements.txt` is empty for now. PyTorch will be added once the benchmark and training code arrive. Note that the correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only), so use the selector on the PyTorch website if the default `pip install torch` does not match your machine.

To build the sizing program (requires [Rust](https://www.rust-lang.org/tools/install)):

```bash
cd sizer_core
cargo build --release
```

Then run the detector from the project root:

```bash
python architecture.py
```

## Requirements

- Python 3.x
- Rust toolchain (for `sizer_core`)
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`amd-smi` / `rocm-smi` / `rocminfo`), depending on hardware. CPU-only systems fall back automatically.

## License

TBD
