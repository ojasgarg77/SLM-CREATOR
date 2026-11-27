# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort, with no fixed deadline). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code. Design decisions and dropped ideas, with reasons, are logged in [docs/decisions.md](docs/decisions.md). Tokenizer test code and results are in [docs/tokenizer_tests_results.md](docs/tokenizer_tests_results.md).

I do not own a GPU for my primary development environment. The NVIDIA GPU code paths have been successfully verified live, while modern and legacy AMD GPU paths rely on AI-described outputs and community testing. Real training and benchmarking are tested on a laptop with an RTX 3050 (6 GB VRAM).

## Planned pipeline

1. **Hardware detection**: NVIDIA, modern or legacy AMD, or CPU-only fallback.
2. **Sizing**: `sizer_core` (Rust) computes a feasible parameter count as the smallest of four limits:
   - **Memory**: parameters that fit in VRAM (about 20 bytes each: fp32 weights, gradients, two Adam states, a little for autocast copies; 10% cushion; activations not included).
   - **Time**: Chinchilla rule (tokens = 20 x parameters), 6 FLOPs per parameter per token, and an efficiency factor on the measured bf16 matmul speed. Parameters scale with the square root of FLOPs, so a 2x error in the factor changes the result by only about 1.4x.
   - **Data**: tokens x 4 epochs / 20.
   - **Cap**: 100M.
3. **Specialization matching**: an LLM matches the user's topic to a pre-curated dataset (skipped for v1; the target is fixed).
4. **Data ingestion**: Python collection and cleaning, then a Rust pipeline (`tokenizer_core`): the tokenizer trains BPE, and the ingestion step encodes the corpus into `train.bin` / `val.bin`. Batching and streaming into training are planned in Python (memory-mapping those files) rather than in Rust.
5. **Training**: plain backpropagation in bf16 mixed precision (fp32 master weights and optimizer state).
6. **Output**: a small local model, with KV-cache planned for inference.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects the GPU (name, VRAM in MiB, index), lets the user pick one if several exist, and asks for the training-time budget in whole hours. Tested on an RTX 3050 laptop GPU; AMD paths are open for community testing. On a CPU-only machine the script currently stops after printing a notice: benchmarking and sizing are skipped, so there is no CPU sizing path yet. Devices reporting under 1024 MiB of VRAM are ignored to filter out integrated graphics, and vendor CLI queries enforce a 10-second timeout.
- ✅ **TFLOPS benchmark** (`architecture.py`): 4096x4096 matmul on the selected GPU, in bf16 when natively supported (otherwise fp32, and it says so), after a 12-second thermal warmup. Reports the median of 10 runs. Matches the vendor tool's GPU name to PyTorch's device so the right card is benchmarked. Measured on the RTX 3050: native bf16, median about 8.9 TFLOPS.
- ✅ **Sizing** (`sizer_core`): Python sends four values over stdin, one per line (VRAM MiB, hours, TFLOPS, training tokens); Rust prints the smallest limit. The token count must be an integer (a value like `12286188.0` makes the sizer exit with an error). A zero token count is an error. The efficiency factor is a placeholder of 0.3 (realistic range 0.1-0.35; plan to use about 0.2 until measured). With these settings a 24-hour run allows roughly 44M parameters by time (hand calculation; confirm by running the sizer).
- 🔧 **Data** (`download_arxiv_text.py`, `data_extractor.py`):
  - **Current corpus** (re-downloaded in the new format): **12,714,236 estimated tokens** (about 44.5M characters at 3.5 characters per token), which gives a data limit of about **2.5M parameters**. Data is the binding constraint. The paper count is not recorded yet. The estimate is about to be replaced by the real count from `train.bin` + `val.bin` (see Tokenizer).
  - The downloader fetches the requested number of **new** papers from arXiv (categories `cs.LG`, `cs.NE`, `stat.ML`, neural-network terms, excluding clinical, survey, medical and forecasting), reads them lazily, skips existing files, and uses ar5iv HTML. Output per paper: `[title]`, `[abstract]`, `[body]`, with `[eq]` before display equations, `[proof]` before proofs, `[section]` before top-level section titles, `[cite]` for citations, and math kept as LaTeX in `$...$`. Papers under 500 cleaned characters are skipped. Prose is lowercased and Unicode-normalized; authors, bibliographies, footnotes, figures and equation-free tables are dropped.
  - **Math cleaning:** font commands (`\mathrm`, `\mathbf`, ...) are removed together with their matching brace (found by counting braces, so nesting is safe), and whitespace inside math is removed except after a backslash-command followed by a letter. Not yet checked: `\text{...}` loses its spaces, escaped braces `\{` `\}` are counted as real braces, and the abstract and title are lowercased with raw LaTeX uncleaned. These functions have no tests yet; try `\mathbf{W_{1}}`, `\mathrm{d}^{2}`, `\mathbf{\mathrm{x}}`, `\beta{x}`, `\sum i` and an unclosed `\mathbf{W`, and check that braces stay balanced.
  - **Token estimate:** `data_extractor.py` divides the character count by 3.5 (from the tokenizer's 10-paper check, which measured bytes per token on papers it was trained on). Units differ (characters, no newlines, versus bytes), so it is approximate. Once the real count is known, it should replace this estimate in the sizer input.
  - **To do:** pick a target size (tokens = 5 x parameters, so about 100M tokens, roughly 350 MB of text, for 20M parameters; a 24-hour run supports about 44M, so more has no use yet); run a few-hundred-paper batch to measure average cleaned size and failure rate, then plan the full download; settle the licensing question before a bulk download (the dataset stays out of the repository); add a stronger partial-conversion check than the 500-character minimum (for example the Adam paper, 1412.6980); print an end-of-run summary and stop after repeated network errors; add a second source (textbooks, lecture notes) only if arXiv runs short.
- 🔧 **Tokenizer** (`tokenizer_core`, Rust, one file `src/main.rs`): byte-level BPE trained from scratch.
  - **IDs** (`u16`): 0-255 raw bytes; specials `[cite]` 256, `<|endoftext|>` 257, `[title]` 258, `[abstract]` 259, `[body]` 260, `[section]` 261, `[eq]` 262, `[proof]` 263; merges from 264, in the order learned. `NET_MERGES` = 8,952, so the **vocabulary is 9,216**.
  - **Special-token table:** `SPECIAL_TOKENS` is the single list the IDs come from (ID = 256 + position). `bpe`, `encode`, `build_vocab` and the first free merge ID all derive from it. **Rule: only append to this list.** Inserting or reordering tags changes the IDs of existing tags without changing `merges.bin`, so nothing would notice. The two regexes in `split_text` still spell the tags out by hand and are the one remaining copy.
  - **Training:** text is split with a regex into specials, `$`, words, whitespace runs and punctuation; merges never cross chunk boundaries. Each distinct chunk is stored once with a count, pair counts are computed once and updated incrementally after each merge, and ties go to the smallest `(left, right)` ID pair, so results do not depend on visiting order. Merges are saved to `merges.bin` (6 bytes each: left, right, new ID, little-endian `u16`; not committed).
  - **Loader check:** `loader` asserts that the file size is a multiple of 6, that the first merged ID is the first free ID (256 + number of specials), that IDs are consecutive, and that both children of each merge are smaller than the new ID. A `merges.bin` from before a change to the special-token list now fails immediately. It does not check the merge count against `NET_MERGES`.
  - **Encode/decode:** encode splits the same way, maps specials directly, and replays merges earliest-learned first (chunks are encoded in parallel with `rayon`); unseen text falls back to raw bytes. The merge lookup table is built once when the `Tokenizer` is created. Decode joins the bytes of each token (unknown ID is an error; invalid UTF-8 becomes a replacement character).
  - **Results** (with 8,952 merges; details in the results file; the test code lives in the results file, not in `main.rs`, and those results predate the shared special-token table, so re-run them):
    - Lossless round trip on a sample with all specials, on edge cases (adjacent and malformed tags, empty and whitespace-only strings, nested-brace math, Unicode) and on 10 whole papers. Special-token counts match everywhere and all IDs are in range.
    - Incremental pair updates match a full recount across 40 merges on a toy corpus.
    - 10 papers: 3.50 bytes per token (optimistic: the tokenizer trained on those papers, the files are not a random sample, and there is no baseline yet).
    - **Merge frequency curve** (winning pair's frequency): 8,372 at merge 500; 1,273 at 2,000; 409 at 4,000; 106 at 8,000; **84 at the last merge**. About 84 x 4 epochs is roughly 330 sightings for the rarest tokens, a reasonable level.
    - Determinism (two fresh runs gave byte-identical `merges.bin`) was confirmed on an earlier corpus and tokenizer version; repeat on the current setup.
    - A round trip passes even if the merges are wrong, so it says nothing about merge correctness. The recount test compares a copy of the merge loop on a toy corpus, not `bpe()` itself.
  - **To do, most important first:**
    1. Derive both regexes in `split_text` from `SPECIAL_TOKENS` (escape each entry, join with `|`) so the table is the only copy.
    2. Move the tokenizer tests into `#[cfg(test)]` so they run with `cargo test`, and add one that loops over `SPECIAL_TOKENS` and checks each entry encodes to exactly its own ID and decodes back. Make the recount test call the same merge-step function as `bpe()`, compare full pair-count maps every few steps, and run it once on a real paper sample.
    3. Repeat the determinism check; compare vocabulary sizes by loading the first N merges of the same file (valid because each merge depends only on earlier ones); measure bytes per token on held-out (val) papers and against an existing tokenizer.
  - **Open decisions:** vocabulary size (9,216 now; the embedding table, vocabulary x model width, takes a large share of a roughly 2.5M-parameter budget: about 1.2M at width 128, 0.6M at width 64, so revisit once the width is chosen); whether a space should attach to the following word (currently separate chunks). Not done: the program uses relative paths and must run from inside `tokenizer_core`.
- 🔧 **Rust ingestion pipeline** (`tokenizer_core/src/main.rs`: `create_dataset`, `write_tokens`, and the encoding step in `main`): turns the cleaned arXiv papers into the two token files the training loop reads. It uses the `Tokenizer` above but is a separate stage.
  - **Split:** paper paths are sorted by file name, so the split is the same on every machine. The first 95% of papers (by count) are train and the remaining 5% are val. The split is by whole paper, never by cutting the token stream, so no paper is half in train and half in val. Because arXiv IDs are roughly chronological, val is the newest papers; if that turns out to skew the val loss, switch to every 20th paper.
  - **Format:** each paper is followed by `<|endoftext|>\n\n\n`, the same format the tokenizer trained on. `train.bin` and `val.bin` are flat sequences of little-endian `u16` IDs (2 bytes per token), so **file size / 2 = token count**.
  - **Stale-file trap:** `train.bin` and `val.bin` are only valid for the exact `merges.bin` that produced them. If both exist, encoding is skipped. Whenever you delete `merges.bin` (new data, changed special tokens, changed `NET_MERGES`), **delete `train.bin` and `val.bin` too**, or they will silently hold wrong IDs.
  - **To do:** record the train and val token counts; check that each file's size is twice its count; decode the first ~200 IDs of `train.bin` (should start with `[title]` and read like a paper) and the last few IDs of `val.bin` (should end with 257 and the newline tokens); put the real total into the sizer in place of the 3.5 estimate. Later stages (batching and streaming) are planned in Python, see below.
- ⏳ **Next: batching, model, training loop.** Not started:
  1. **Batching (Python):** memory-map `train.bin` as little-endian `uint16`, take a random window of tokens as `x` and the same window shifted by one as `y`.
  2. **Model:** token embedding, causal self-attention, MLP, layer norm, residual connections, output layer tied to the embedding table.
  3. **Training loop:** bf16 autocast, AdamW, learning-rate warmup and decay, periodic loss on `val.bin`.
  4. Dataset specialization matching and inference with KV-cache come later.

## Measuring the efficiency factor

The factor answers: "of the speed a pure matmul benchmark reaches, how much do I keep in real training?" Once the training loop exists, measure it: tokens per second x 6 x parameters (about 8 x with gradient checkpointing) gives achieved FLOPs per second; divide by the benchmark TFLOPS (same precision). Run long enough for thermal throttling to show, then put the result in `sizer_core/src/main.rs`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

VS Code / Cursor users: A .vscode/settings.json is included in the repository so new integrated terminals automatically activate .venv once created.

`requirements.txt` has `torch`, `arxiv`, `requests` and `beautifulsoup4`. The right PyTorch build depends on your hardware, so use the selector on the PyTorch website if the default `pip install torch` does not match. (architecture.py only imports torch after a dedicated GPU is detected, so hardware scanning works even before PyTorch is installed.)

## Building the Rust programs

Requires [Rust](https://www.rust-lang.org/tools/install), a recent stable toolchain (the code uses let-chains and `as_chunks`, so roughly 1.88 or newer). Rebuild with `--release` after any Rust change, because `architecture.py` runs the release binary of `sizer_core`:

```bash
cd sizer_core
cargo build --release
cd ../tokenizer_core
cargo build --release     # needs the regex, indexmap and rayon crates (downloaded on first build)
```

The `dataset/` folder is not in the repository (it is in `.gitignore`: large files, unchecked copyright). It contains `arxiv_texts/`, created by `download_arxiv_text.py`. `merges.bin`, `train.bin` and `val.bin` are generated output and are not committed either.

## Running

From the project root:

```bash
python download_arxiv_text.py     # downloads and cleans arXiv papers (needs internet)
python architecture.py            # hardware detection, benchmark, token count and sizing
```

The tokenizer is not connected to the Python side yet and runs on its own from inside its folder. One `cargo run --release` does everything in order, skipping any step whose output already exists: it trains BPE and writes `merges.bin`, loads and checks it, then encodes the papers into `train.bin` and `val.bin` and prints the token counts. **Delete `merges.bin`, `train.bin` and `val.bin` whenever the dataset, the special-token list or `NET_MERGES` changes**, otherwise the run only loads the old files:

```bash
cd tokenizer_core
cargo run --release
```

## Requirements

- Python 3.x and the Rust toolchain
- GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tools (`amd-smi` / `rocm-smi` / `rocminfo`); without a supported GPU the script reports it and stops before sizing
- Intended training setup: a GPU with native bf16 (NVIDIA Ampere or newer, such as the RTX 3050)
- Internet access for the download script

## Contributing

Contributions are welcome! Since the NVIDIA paths are validated, community testing now focuses on **AMD GPUs**. If you have one, run the architecture script and report how it handles your system. Tracking issue: **[Issue #1: Help Needed: Live testing & validation of GPU detection paths (NVIDIA/AMD)](https://github.com/ojasgarg77/SLM-CREATOR/issues/1#issue-6224761352)**. For other bugs, documentation fixes or dataset cleaning improvements, open an issue or a pull request.

## License

Copyright 2026 ojasgarg77

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://apache.org](http://apache.org) or in the [LICENSE](LICENSE) file in this repository.
