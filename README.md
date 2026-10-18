# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code.

I do not own a GPU. The GPU code paths (NVIDIA, modern AMD, legacy AMD) cannot be run on my machine, so for GPU edge cases I rely on AI-described outputs (for example, what the vendor tools would print on various hardware) rather than AI-written code, and I test my parsing against mock tool output.

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name and total VRAM.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware and the user's training-time budget and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic, not an LLM guess. It takes the smallest of the limits:
   - **Memory limit**: how many parameters fit in VRAM (about 20 bytes per parameter, with a 10% cushion).
   - **Time limit**: how many parameters can be trained in the time budget, using the Chinchilla rule (training tokens = 20 x parameters), 6 FLOPs per parameter per token, and a utilization factor of 0.3 because real training is slower than a pure matmul benchmark.
   - **Data limit** (planned): how many parameters the available training data can support (tokens x epochs / 20).
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick. (For v1 the target is fixed to neural networks, so this step can wait.)
4. **Data ingestion**: PDF/text extraction and cleaning in Python, then a Rust pipeline that handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1. A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. If several GPUs are found, the user picks one. Also asks for the training-time budget in hours. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.
- ✅ **TFLOPS benchmark** (`architecture.py`): measures the machine's real speed with a matrix-multiplication benchmark (skipped on CPU-only systems).
- 🔧 **Sizing** (`sizer_core`, Rust): memory and time limits work, with a 100M parameter cap.
  - Python sends three values to the Rust program over standard input, one per line: VRAM (MiB), training hours, and measured TFLOPS.
  - Rust parses them, computes the memory and time limits, and prints the smallest of those and the cap to standard output.
  - Python converts the result to an integer, prints it, and says so when the 100M cap is what limited the model.
  - Two failure cases are handled separately: if the sizer binary is missing, Python tells the user to set up Rust and `sizer_core` and waits before retrying; if the sizer runs but fails, Python shows its exit code and error output and stops, since retrying the same input would fail the same way.
  - In progress: the data limit (a fourth input line holding the training-data token count). A stand-in token count will be used until the dataset is extracted.
  - To do: split reading from computing in Rust so the math can be unit-tested, validate the values Python sends, handle a failed sizer run cleanly, and use the returned parameter count on the Python side.
- 🔧 **Data extraction** (`data_extractor.py`): finds every PDF in the `dataset/` folder, extracts the text of all pages with PyMuPDF, fixes typographic ligatures (such as "ﬁ" to "fi"), removes footnote markers, and prints a character count per file.
  - To do: keep paragraph breaks while joining layout lines (block-based extraction), handle words split across lines by hyphens, drop junk blocks such as page numbers, and convert character counts to token estimates for the sizer.
- ⏳ Dataset specialization matching: not started.
- ⏳ Collecting the full neural-networks dataset: not started (the 7 PDFs in `dataset/` are a small test set for the extraction pipeline).
- ⏳ Rust ingestion pipeline (tokenizing, batching, streaming): not started.
- ⏳ Training loop: not started.
- ⏳ Inference with KV-cache: not started.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` contains `torch`, which the TFLOPS benchmark needs, and `pymupdf`, which the data extractor needs. The correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only), so use the selector on the PyTorch website if the default `pip install torch` does not match your machine.

To build the sizing program (requires [Rust]):

```bash
cd sizer_core
cargo build --release
```

Create a `dataset/` folder in the project root and put the PDFs you want to train on inside it. The folder is not part of the repository (it is listed in `.gitignore`), because the PDFs are copyrighted material and large. Without it, the extractor prints "THE DIRECTORY DOES NOT EXIST."

Then run the tools from the project root:

```bash
python architecture.py      # hardware detection, benchmark and sizing
python data_extractor.py    # PDF text extraction and character counts
```

## Requirements

- Python 3.x
- Rust toolchain (for `sizer_core`)
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`amd-smi` / `rocm-smi` / `rocminfo`), depending on hardware. CPU-only systems fall back automatically.

## License

TBD
