# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code.

I do not own a GPU. The GPU code paths (NVIDIA, modern AMD, legacy AMD) cannot be run on my machine, so for GPU edge cases I rely on AI-described outputs (for example, what the vendor tools would print on various hardware) rather than AI-written code, and I test my parsing against mock tool output.

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name and total VRAM.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware and the user's training-time budget and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic, not an LLM guess. It takes the smallest of three limits:
   - **Memory limit**: how many parameters fit in VRAM.
   - **Time limit**: how many parameters can be trained in the time budget, using the Chinchilla rule (training tokens = 20 x parameters) and 6 FLOPs per parameter per token.
   - **Data limit** (planned): how many parameters the available training data can support (tokens / 20).
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick. (For v1 the target is fixed to neural networks, so this step can wait.)
4. **Data ingestion**: a Rust pipeline handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1. A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. If several GPUs are found, the user picks one. Also asks for the training-time budget in hours. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.
- ✅ **TFLOPS benchmark** (`architecture.py`): measures the machine's real speed with a matrix-multiplication benchmark (skipped on CPU-only systems).
- 🔧 **Sizing** (`sizer_core`, Rust): memory and time limits work.
  - Python sends three values to the Rust program over standard input, one per line: VRAM (MiB), training hours, and measured TFLOPS.
  - Rust parses them, computes the memory limit and the time limit, and prints the smaller one to standard output.
  - In progress: the data limit (a fourth input line holding the training-data token count), so the final answer is the smallest of three limits. A stand-in token count will be used until the dataset is extracted.
  - To do: use the returned parameter count on the Python side.
- ⏳ Dataset specialization matching: not started.
- ⏳ Data collection and extraction for the neural-networks dataset: not started.
- ⏳ Rust ingestion pipeline: not started.
- ⏳ Training loop: not started.
- ⏳ Inference with KV-cache: not started.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` currently contains `torch`, which the TFLOPS benchmark needs. The correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only), so use the selector on the PyTorch website if the default `pip install torch` does not match your machine.

To build the sizing program (requires [Rust]):

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
