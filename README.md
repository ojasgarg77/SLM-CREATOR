# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort, with no fixed deadline). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code. Design decisions and dropped ideas, with reasons, are logged in [docs/decisions.md](docs/decisions.md).

I do not own a GPU. The GPU code paths (NVIDIA, modern AMD, legacy AMD) cannot be run on my machine, so for GPU edge cases I rely on AI-described outputs (for example, what the vendor tools would print on various hardware) rather than AI-written code, and I test my parsing against mock tool output. Real training is planned on a laptop with an RTX 4050 (6 GB VRAM).

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name, total VRAM and device index.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware, the user's training-time budget and the size of the training data, and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic, not an LLM guess. It takes the smallest of four limits:
   - **Memory limit**: how many parameters fit in VRAM (about 20 bytes per parameter, with a 10% cushion). The 20 bytes cover fp32 master weights, gradients and Adam's two states, plus a little for autocast's temporary bf16 weight copies. Activations are *not* included; they depend on batch size and context length and must fit in the leftover VRAM.
   - **Time limit**: how many parameters can be trained in the time budget, using the Chinchilla rule (training tokens = 20 x parameters), 6 FLOPs per parameter per token, and an efficiency factor applied to the measured bf16 matmul speed. Because parameters scale with the square root of FLOPs, an error of 2x in this factor changes the parameter count by only about 1.4x.
   - **Data limit**: how many parameters the available training data can support: tokens x 4 epochs / 20.
   - **Cap**: 100M parameters.
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick. (For v1 the target is fixed to neural networks, so this step can wait.)
4. **Data ingestion**: text collection, extraction and cleaning in Python, then a Rust pipeline that handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1, in **bf16 mixed precision** (matmuls in bf16 on the tensor cores; master weights, optimizer state and numerically sensitive operations in fp32). A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. Each detected GPU is recorded as (architecture, VRAM in MiB, device index). If several GPUs are found, the user picks one, and the chosen device index is passed on to the benchmark. Also asks for the training-time budget in hours. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.

- 🔧 **TFLOPS benchmark** (`architecture.py`): measures the machine's real matmul speed with a 4096x4096 benchmark on the selected GPU (skipped on CPU-only systems).
  - Runs in **bf16** when `torch.cuda.is_bf16_supported()` is true, otherwise falls back to fp32 and says so. The benchmark precision must match the training precision, otherwise the efficiency factor is meaningless.
  - Does 10 warmup multiplications, then times several runs of 20 back-to-back multiplications each, with `torch.cuda.synchronize` before and after each timed window.
  - Not yet run on the target GPU.
  - To do: the benchmark is only about a second long, so it measures boost clocks, not the sustained speed of a multi-hour laptop run. 
  - To do: `is_bf16_supported()` can return true for slow emulated bf16 on some older cards; check the installed PyTorch version for an option to exclude emulation.
  - To do: confirm that the device index reported by the vendor tool matches the index PyTorch uses.

- ✅ **Sizing** (`sizer_core`, Rust): memory, time and data limits all work, with a 100M parameter cap.
  - Python sends four values to the Rust program over standard input, one per line: VRAM (MiB), training hours, measured TFLOPS, and the total number of training tokens. The Rust side does not care which precision produced the TFLOPS number.
  - Rust parses them, computes the four limits, and prints the smallest to standard output. A token count of zero is an error, so a missing dataset cannot silently produce a zero-parameter model.
  - Python converts the result to an integer, prints it, and says so when the 100M cap is what limited the model.
  - Two failure cases are handled separately: if the sizer binary is missing, Python tells the user to set up Rust, `sizer_core` and the dataset, and waits before retrying; if the sizer runs but fails, Python shows its exit code and error output and stops, since retrying the same input would fail the same way.
  - The efficiency factor is currently 0.3 in `main.rs`. It is a placeholder, not a measurement. Against a bf16 baseline the realistic range is about 0.1-0.35, and I plan to lower it to about 0.2 until I can measure it (see "Measuring the efficiency factor" below).

- 🔧 **Data collection and extraction**:
  - `download_english_data.py`: streams the first 500,000 sentences of a cleaned BookCorpus dataset from Hugging Face into `dataset/english_dataset/english_dataset.txt` (about 41.5M characters). It skips the download if the file already exists. This is general English fiction, already lowercase, and is meant as a base for English, not as neural-network content.
  - `download_arxiv_text.py`: interactively prompts the user for the number of papers to download, searches arXiv for papers matching a specific NLP/ML/LLM query string, fetches each paper's ar5iv HTML page, and writes structured text files to `dataset/arxiv_texts/<arxiv_id>.txt` with fields: `title--`, `arxiv id--`, `abstract--`, and `body--`. Cleaning and processing steps include:
    - Lowercasing non-math content while protecting LaTeX math blocks.
    - Extracting and cleaning math output (`clean_math_output`) to keep expressions in `$` LaTeX delimiters using HTML annotations or fallback text.
    - Cleaning paper text (`clean_paper_text`) by stripping invisible Unicode characters, normalizing spacing around punctuation, handling hyphenated line breaks, and collapsing excessive newlines.
    - Replacing in-text citations with a `[cite]` placeholder.
    - Filtering out author info, affiliations, bibliographies, footnotes, figures, and irrelevant tables.
    - Skipping existing files and handling/deleting corrupt or empty files.
  - `data_extractor.py`: scans PDFs in `dataset/pdfs/`, text files in `dataset/arxiv_texts/`, and text files in `dataset/english_dataset/`. It extracts text from PDFs using PyMuPDF, cleans typographic ligatures (such as "ﬀ" to "ff", "ﬁ" to "fi", etc.) using `LIGATURE_AND_UNICODE_DICTIONARY`, counts characters across lines for all sources, and estimates total tokens (`main()`) at 4 characters per token to pass back to `architecture.py`. The 4 characters per token figure is a rough guess until the real tokenizer exists.
  - To do (arXiv):
    - Check why some papers (for example Adam, 1412.6980) appear to convert only partially, and add a check for partial pages.
    - Re-run the table rule on more papers to confirm it.
    - Decide what mentions like "Eq. 2" should become now that equation labels are dropped (for example a placeholder token, like `[cite]`).
    - Normalize LaTeX style differences (`\it`, `\bf`, `\mathrm`) using frequency counts over many papers.
  - To do (cleaning, PDFs and English data): normalize the spacing around punctuation in the English data, drop ebook boilerplate and URL lines, share one cleaning function between all sources, keep paragraph breaks in PDFs, and handle words split across lines by hyphens.
  - To do (data): collect much more on-topic neural-network text, because on-topic text is only a small share of the current mix and, once the benchmark is fixed, the data limit is likely to be the binding one. Cleaned arXiv papers are much shorter than raw PDFs, so the number of papers needed is likely higher than first estimated; measure tokens per usable paper on a sample.
- 🔧 Tokenizer (must keep `[cite]` and `$` as single tokens and preserve whitespace exactly): (added tokenize_core--WIP).
- ⏳ Dataset specialization matching: not started.
- ⏳ Rust ingestion pipeline (tokenizing, batching, streaming): not started.
- ⏳ Training loop (bf16 mixed precision): not started.
- ⏳ Inference with KV-cache: not started.

## Measuring the efficiency factor

The efficiency factor answers: "of the speed a pure matmul benchmark reaches, how much do I keep in real training?" Real training loses time to memory-bound operations (layer norm, softmax, optimizer step), small matrices, kernel-launch gaps, data loading, and laptop thermal throttling. Once the training loop exists, measure it instead of guessing:

1. Count the tokens per second the training loop processes.
2. Multiply by 6 x the parameter count to get the FLOPs per second actually achieved (about 8 x if gradient checkpointing is used).
3. Divide by the benchmark TFLOPS (measured in the same precision).
4. Run for long enough that thermal throttling shows up, then put the result into `main.rs`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` contains `torch` (the TFLOPS benchmark), `pymupdf` (PDF extraction), `datasets` and `python-dotenv` (the English download), `arxiv` and `requests` (finding and fetching arXiv papers), and `beautifulsoup4` (parsing the ar5iv HTML). The correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only), so use the selector on the PyTorch website if the default `pip install torch` does not match your machine.

To build the sizing program (requires [Rust]):

```bash
cd sizer_core
cargo build --release
```

To build the tokenizer program (empty right now):

```bash
cd tokenizer_core
cargo build --release
```

Rebuild with `--release` after any change to the Rust code, because `architecture.py` runs the release binary(sizer_core).

### Dataset folder

The `dataset/` folder is not part of the repository (it is listed in `.gitignore`), because it holds large text files and unchecked copyright data. It looks like this:

```
dataset/
  pdfs/              PDFs you add by hand
  arxiv_texts/       created by download_arxiv_text.py
  english_dataset/   created by download_english_data.py
```

Without `dataset/pdfs/`, the extractor prints "THE DIRECTORY DOES NOT EXIST." and counts only the text files.

### Running

Run everything from the project root:

```bash
python download_english_data.py   # one-time English text download (needs internet)
python download_arxiv_text.py     # downloads and cleans arXiv papers as text (needs internet)
python architecture.py            # hardware detection, benchmark, token count and sizing
```

## Requirements

- Python 3.x
- Rust toolchain (for `sizer_core`)
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`amd-smi` / `rocm-smi` / `rocminfo`), depending on hardware. CPU-only systems fall back automatically.
- For the intended training setup: a GPU with bf16 support (NVIDIA Ampere or newer, such as the RTX 4050).
- Internet access for the download scripts

## License

TBD