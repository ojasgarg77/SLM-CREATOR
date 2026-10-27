# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort, with no fixed deadline). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code. Design decisions and dropped ideas, with reasons, are logged in [docs/decisions.md](docs/decisions.md).

I do not own a GPU. The GPU code paths (NVIDIA, modern AMD, legacy AMD) cannot be run on my machine, so for GPU edge cases I rely on AI-described outputs (for example, what the vendor tools would print on various hardware) rather than AI-written code, and I test my parsing against mock tool output. Real training is planned on a laptop with an RTX 4050 (6 GB VRAM).

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name and total VRAM.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware, the user's training-time budget and the size of the training data, and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic, not an LLM guess. It takes the smallest of four limits:
   - **Memory limit**: how many parameters fit in VRAM (about 20 bytes per parameter, with a 10% cushion).
   - **Time limit**: how many parameters can be trained in the time budget, using the Chinchilla rule (training tokens = 20 x parameters), 6 FLOPs per parameter per token, and a utilization factor of 0.3 (an estimate, not measured) because real training is slower than a pure matmul benchmark.
   - **Data limit**: how many parameters the available training data can support: tokens x 4 epochs / 20.
   - **Cap**: 100M parameters.
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick. (For v1 the target is fixed to neural networks, so this step can wait.)
4. **Data ingestion**: text collection, extraction and cleaning in Python, then a Rust pipeline that handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1. A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. If several GPUs are found, the user picks one. Also asks for the training-time budget in hours. Tested live on a CPU-only machine; the GPU paths have been tested against mock tool output only.

- ✅ **TFLOPS benchmark** (`architecture.py`): measures the machine's real speed with an fp32 4096x4096 matrix-multiplication benchmark (skipped on CPU-only systems). Not yet run on the target GPU.

- ✅ **Sizing** (`sizer_core`, Rust): memory, time and data limits all work, with a 100M parameter cap.
  - Python sends four values to the Rust program over standard input, one per line: VRAM (MiB), training hours, measured TFLOPS, and the total number of training tokens.
  - Rust parses them, computes the four limits, and prints the smallest to standard output. A token count of zero is an error, so a missing dataset cannot silently produce a zero-parameter model.
  - Python converts the result to an integer, prints it, and says so when the 100M cap is what limited the model.
  - Two failure cases are handled separately: if the sizer binary is missing, Python tells the user to set up Rust, `sizer_core` and the dataset, and waits before retrying; if the sizer runs but fails, Python shows its exit code and error output and stops, since retrying the same input would fail the same way.

- 🔧 **Data collection and extraction**:
  - `download_english_data.py`: streams the first 500,000 sentences of a cleaned BookCorpus dataset from Hugging Face into `dataset/english_dataset/english_dataset.txt` (about 41.5M characters). It skips the download if the file already exists. This is general English fiction, already lowercase, and is meant as a base for English, not as neural-network content.
  - `download_arxiv_text.py`: searches arXiv for papers matching a query (currently `cs.LG` papers about "neural networks", 3 results by default), fetches each paper's ar5iv HTML page, and writes cleaned text to `dataset/arxiv_texts/<arxiv_id>.txt` with TITLE / ARXIV ID / ABSTRACT / BODY fields. Cleaning rules:
    - Math is kept as `$LaTeX$`, taken from the HTML annotation element.
    - Citations become `[cite]`.
    - Section numbers are kept; equation labels like `(2)` are dropped.
    - Authors, affiliations, the bibliography, footnotes and figures are removed.
    - Tables are removed unless they contain equation rows (some papers lay out multi-line equations inside a table).
    - Skips files that already exist and deletes corrupt or empty ones.
    - Tested on a few papers only; not yet scaled up.
  - `data_extractor.py`: finds every PDF in `dataset/pdfs/` and every text file in `dataset/english_dataset/`, extracts the PDF text with PyMuPDF, fixes typographic ligatures (such as "ﬁ" to "fi"), removes footnote markers, counts characters, and estimates tokens at 4 characters per token. `main()` returns the total so `architecture.py` can import it. 
  - To do (arXiv):
    - Check why some papers (for example Adam, 1412.6980) appear to convert only partially, and add a check for partial pages.
    - Re-run the table rule on more papers to confirm it.
    - Decide what mentions like "Eq. 2" should become now that equation labels are dropped (for example a placeholder token, like `[CITE]`).
    - Normalize LaTeX style differences (`\it`, `\bf`, `\mathrm`) using frequency counts over many papers.
  - To do (cleaning, PDFs and English data): normalize the spacing around punctuation in the English data, drop ebook boilerplate and URL lines, share one cleaning function between all sources, keep paragraph breaks in PDFs, and handle words split across lines by hyphens.
  - To do (data): collect much more on-topic neural-network text, because on-topic text is only a small share of the current mix. Cleaned arXiv papers are much shorter than raw PDFs, so the number of papers needed is likely higher than first estimated; measure tokens per usable paper on a sample.
- ⏳ Tokenizer (must keep `[cite]` and `$` as single tokens and preserve whitespace exactly): not started.
- ⏳ Dataset specialization matching: not started.
- ⏳ Rust ingestion pipeline (tokenizing, batching, streaming): not started.
- ⏳ Training loop: not started.
- ⏳ Inference with KV-cache: not started.

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

Rebuild with `--release` after any change to the Rust code, because `architecture.py` runs the release binary.

### Dataset folder

The `dataset/` folder is not part of the repository (it is listed in `.gitignore`), because it holds copyrighted PDFs and large text files. It looks like this:

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
- Internet access for the download scripts

## License

TBD
