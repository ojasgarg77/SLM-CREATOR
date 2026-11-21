# SLM Creation

A from-scratch, locally-trainable, hyperspecialized small language model (SLM) trainer.

This project trains a small text-only language model **from scratch** (random weight initialization, no fine-tuning of existing pretrained models) in PyTorch. The model is sized to fit what the user's own hardware can realistically train, capped at 100M parameters.

The v1 target is a model hyperspecialized in **neural networks**, built for my own use.

## About this project

This is an open-ended personal learning project (roughly a 5-6 month effort, with no fixed deadline). I am learning by building: **the entire codebase is handwritten by me**, piece by piece. AI is used as a tutor for explanations and concepts, not as a source of finished code. Design decisions and dropped ideas, with reasons, are logged in [docs/decisions.md](docs/decisions.md).

I do not own a GPU for my primary development environment. The NVIDIA GPU code paths have been successfully verified live, while modern and legacy AMD GPU paths rely on AI-described outputs and community testing. Real training and benchmarking are tested on a laptop with an RTX 3050 (6 GB VRAM).

## Planned pipeline

1. **Hardware detection**: detect an NVIDIA GPU, a modern or legacy AMD GPU, or fall back to CPU-only. Collect GPU name, total VRAM and device index.
2. **Sizing**: a Rust program (`sizer_core`) takes the detected hardware, the user's training-time budget and the size of the training data, and works out a feasible parameter count (capped at 100M). Sizing is deterministic arithmetic, not an LLM guess. It takes the smallest of four limits:
   - **Memory limit**: how many parameters fit in VRAM (about 20 bytes per parameter, with a 10% cushion). The 20 bytes cover fp32 master weights, gradients and Adam's two states, plus a little for autocast's temporary bf16 weight copies. Activations are *not* included; they depend on batch size and context length and must fit in the leftover VRAM.
   - **Time limit**: how many parameters can be trained in the time budget, using the Chinchilla rule (training tokens = 20 x parameters), 6 FLOPs per parameter per token, and an efficiency factor applied to the measured bf16 matmul speed. Because parameters scale with the square root of FLOPs, an error of 2x in this factor changes the parameter count by only about 1.4x.
   - **Data limit**: how many parameters the available training data can support: tokens x 4 epochs / 20.
   - **Cap**: 100M parameters.
3. **Specialization matching**: the user describes what they want the model specialized in, and an LLM matches that to one of several pre-curated datasets. The user can override the pick. (For v1 the target is fixed to neural networks, so this step can wait.)
4. **Data ingestion**: text collection, extraction and cleaning in Python, then a Rust pipeline (`tokenizer_core` and later stages) that handles tokenizing, batching, and streaming training data into the PyTorch training loop.
5. **Training**: plain backpropagation for v1, in **bf16 mixed precision** (matmuls in bf16 on the tensor cores; master weights, optimizer state and numerically sensitive operations in fp32). A Forward-Forward + backprop hybrid is a possible experimental mode much later, not the default.
6. **Output**: a small, locally-runnable, hyperspecialized text model, with KV-cache support planned for efficient conversation-history handling at inference time.

## Current status

- ✅ **Hardware detection** (`architecture.py`): detects NVIDIA (`nvidia-smi`), modern AMD (`amd-smi`), legacy AMD (`rocm-smi`/`rocminfo`), or CPU-only. Each detected GPU is recorded as (architecture, VRAM in MiB, device index, GPU name). If several GPUs are found, the user picks one, and the chosen device index is passed on to the benchmark. Also asks for the training-time budget in hours. Tested live on an RTX 3050 laptop GPU (6144 MiB VRAM); the AMD GPU paths remain open for live community testing.

- ✅ **TFLOPS benchmark** (`architecture.py`): measures the machine's real matmul speed with a 4096x4096 benchmark on the selected GPU (skipped on CPU-only systems).
  - Runs in **bf16** when native hardware bf16 is supported (checking `torch.cuda.is_bf16_supported(including_emulation=False)` on PyTorch 2.4+ or compute capability >= 8.0 on older versions, avoiding slow software-emulated bf16), otherwise falls back to fp32 and says so. The benchmark precision must match the training precision, otherwise the efficiency factor is meaningless.
  - Runs a 12-second thermal warmup soak to settle past laptop boost clocks, then times 10 runs of 20 back-to-back multiplications each, with `torch.cuda.synchronize` before and after each timed window, and reports the **median** TFLOPS instead of the mean.
  - Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before importing PyTorch and matches the vendor tool's GPU name against `torch.cuda.get_device_name()` (`resolve_pytorch_index`) so the `cuda:X` index targets the selected card.
  - Verified live on an RTX 3050 laptop GPU (6144 MiB VRAM, native `bfloat16` support, median ~8.895 TFLOPS).

- ✅ **Sizing** (`sizer_core`, Rust): memory, time and data limits all work, with a 100M parameter cap.
  - Python sends four values to the Rust program over standard input, one per line: VRAM (MiB), training hours, measured TFLOPS, and the total number of training tokens. The token count must be a whole number: the Rust side parses it as an integer and exits with an error on a value like `12286188.0`. The Rust side does not care which precision produced the TFLOPS number.
  - Rust parses them, computes the four limits, and prints the smallest to standard output. A token count of zero is an error, so a missing dataset cannot silently produce a zero-parameter model.
  - Python converts the result to an integer, prints it, and says so when the 100M cap is what limited the model.
  - Two failure cases are handled separately: if the sizer binary is missing, Python tells the user to set up Rust, `sizer_core` and the dataset, and waits before retrying; if the sizer runs but fails, Python shows its exit code and error output and stops, since retrying the same input would fail the same way.
  - The efficiency factor is currently 0.3 in `sizer_core/src/main.rs`. It is a placeholder, not a measurement. Against a bf16 baseline the realistic range is about 0.1-0.35, and I plan to lower it to about 0.2 until I can measure it (see "Measuring the efficiency factor" below).
  - With the benchmark above and the 0.3 placeholder, the time limit for a 24-hour run works out to roughly 44M parameters (hand calculation; confirm by running the sizer).

- 🔧 **Data collection and extraction**:
  - **Current corpus: about 44.3M characters of cleaned arXiv text, estimated at 12,286,188 tokens.** The data limit (tokens x 4 / 20) is therefore about **2.46M parameters**. Every other limit is far higher, so **data is the binding constraint**. The paper count is not recorded yet (see to-do).
  - `download_arxiv_text.py`: interactively prompts the user for the number of papers to download, searches arXiv for papers matching a query string (categories `cs.LG`, `cs.NE` and `stat.ML`, with neural-network terms such as "neural network", "transformer", "attention mechanism" and "gradient descent", excluding clinical, survey, medical and forecasting papers), fetches each paper's ar5iv HTML page, and writes structured text files to `dataset/arxiv_texts/<arxiv_id>.txt` with fields: `title--`, `arxiv id--`, `abstract--`, and `body--`. Cleaning and processing steps include:
    - Lowercasing non-math content while protecting LaTeX math blocks.
    - Normalizing typographic Unicode in prose (not in math) to plain ASCII through `LIGATURE_AND_UNICODE_DICTIONARY` in `process_prose_nodes`: ligatures (`ﬁ` to `fi`), curly quotes, the different dashes and the minus sign to `-`, the ellipsis character to `...`, the many Unicode space characters to a normal space, and zero-width characters and soft hyphens removed.
    - Extracting and cleaning math output (`clean_math_output`) to keep expressions in `$` LaTeX delimiters using HTML annotations or fallback text. Style commands (`\mathrm`, `\mathbf`, `\it`, `\bf`, ...) are stripped and whitespace inside math is removed.
    - Cleaning paper text (`clean_paper_text`) by stripping invisible Unicode characters, normalizing spacing around punctuation, handling hyphenated line breaks, and collapsing excessive newlines.
    - Replacing in-text citations with a `[cite]` placeholder.
    - Filtering out author info, affiliations, bibliographies, footnotes, figures, and irrelevant tables.
    - Skipping existing files and handling/deleting corrupt or empty files.
  - `data_extractor.py`: scans the text files in `dataset/arxiv_texts/`, cleans typographic ligatures using `LIGATURE_AND_UNICODE_DICTIONARY`, counts characters across lines, and estimates total tokens (`main()`) as characters divided by 3.61, converted to an integer, to pass back to `architecture.py`.
    - Where 3.61 comes from: the "first 10 papers" check of the tokenizer (487,146 bytes into 135,092 tokens, with the current merges). It is a *bytes* per token figure from 10 papers that the tokenizer was trained on, while the extractor counts *characters*. The text is mostly ASCII, so characters per token is probably only a percent or two lower. A lower figure means more tokens, so the estimate leans conservative. The old guess of 4 characters per token also under-counted tokens not the other way round.
    - PDF input and the separate English-text download were removed; all text comes from the ar5iv HTML download.
  - **Known bugs in the math cleaning (`clean_math_output`), not fixed yet:**
    - The `\beta` entry in the `\mathrm|\mathit|...` regex is a leftover: `\beta` is a real symbol, not a font command, and must be removed from that list.
    - The font-command pattern stops at the first `}`, so nested braces break it: `\mathbf{W_{1}}` becomes `W_{1` plus a stray `}`. Fix by removing the command, its opening brace and only the matching closing brace (counting braces), and test with `\mathbf{W_{1}}`, `\mathrm{d}^{2}`, `\mathbf{\mathrm{x}}`, `\beta{x}` and an unclosed `\mathbf{W`. Check that braces stay balanced.
    - Removing all whitespace inside math glues a command to the next token (`\sum i` becomes `\sumi`), which changes its meaning. A space after a backslash-command followed by a letter must be kept.
    - After fixing these, the existing papers must be re-downloaded (or re-cleaned), and the tokenizer retrained.
  - To do (arXiv):
    - Check why some papers (for example Adam, 1412.6980) appear to convert only partially, and add a check for partial pages (for example a minimum cleaned size per paper).
    - In the query string, `stat.ML` has no `cat:` prefix, unlike the other two categories, so arXiv may treat it as a plain text term instead of a category filter. Check this before a bulk download.
    - Make reruns cheap. Right now `list(client.results(...))` fetches the whole result list before the loop, and "N papers" means the top N results, so asking for 1,000 when 600 exist re-lists 1,000 to find 400. Planned change: iterate the results lazily, treat the user's number as "N *new* papers" (count only successful saves, skip existing files without counting them), and use a separate, larger limit as a safety cap on how far to look. Test by asking for 5 new papers when most top results already exist: exactly 5 new files must appear.
    - Record the paper count and the average cleaned size per paper.
  - To do (data): collect much more on-topic neural-network text. Plan:
    1. Decide a target size. Working backwards from the Chinchilla rule, tokens = 5 x parameters (4 epochs, 20 tokens per parameter), so about 100M tokens (roughly 360 MB of text) supports about 20M parameters. The current 12.3M tokens is about 8 times too few for that.
    2. Run a batch of a few hundred papers first, and measure the average cleaned size per paper and the failure rate, then plan the full download from those two numbers.
    3. Check the time limit too: a 24-hour run supports roughly 44M parameters at the current benchmark and placeholder factor, so data beyond that has no use yet.
    4. Settle the licensing question before the bulk download (see `docs/decisions.md`): the dataset stays out of the repository, and the question of whether the trained model will be shared needs an answer.
    5. Add a second source type only if arXiv runs short (textbooks, lecture notes, documentation), and keep off-topic text small.
  - To do (tokens per char): replace the 3.61 estimate with a measured number over the whole corpus, using the final tokenizer (held-out papers if possible). The figure must be recomputed whenever the vocabulary size changes, because a smaller vocabulary compresses less.

- 🔧 **Tokenizer** (`tokenizer_core`, Rust, work in progress): a byte-level BPE tokenizer trained from scratch. `[cite]` and `$` must each stay a single chunk, and whitespace must be preserved exactly.
  - **Token IDs** (`u16`, so up to 65,535): 0-255 are the raw bytes, 256 is `[cite]`, 257 is `<|endoftext|>`, and 258 onward are merged tokens numbered in the order they are learned.
  - **Done so far:**
    - Reads every `.txt` file in `dataset/arxiv_texts/` and joins them, with an `<|endoftext|>` marker and blank lines after each paper, so documents are never run together.
    - Splits the text into chunks with a regex (`regex` crate): `<|endoftext|>`, the `[cite]` placeholder, `$`, words, whitespace runs, and single punctuation characters. `[cite]` becomes the single ID 256, `<|endoftext|>` the single ID 257; every other chunk becomes its UTF-8 bytes widened to `u16`. Merges never cross a chunk boundary. The same splitting function is used for training and for encoding, so both see identical chunks.
    - **Stores each distinct chunk once, with a count.** A word like "the" appears thousands of times, but it is only merged once and its pairs are counted once, multiplied by how often the chunk occurs.
    - **Counts pairs once, then updates incrementally.** All adjacent pairs are counted (weighted by chunk count) before the first merge. After each merge, only the chunks that contain the winning pair are touched: their old pairs are subtracted from the counts and the pairs of the rewritten chunk are added. The winning pair's own entry is removed.
    - Runs the merge loop: pick the most frequent pair, give it the next free ID, and rewrite the affected chunks left to right without overlapping matches.
    - **Breaks frequency ties with an explicit rule:** the pair with the smallest `(left, right)` token IDs wins (see `docs/decisions.md`). The result no longer depends on the order the chunks or pairs happen to be visited in.
    - **Number of merges:** `NET_MERGES` is currently 16,126, which gives a vocabulary of 16,384 tokens (256 bytes + 2 special tokens + 16,126 merges). It can never exceed what `u16` allows. Progress is printed every 500 iterations.
    - **Saves the learned merges to `tokenizer_core/merges.bin`.** Each merge is one fixed 6-byte record: left token, right token, new token ID, each a little-endian `u16`, in the order the merges were learned. That order matters, because encoding must replay the merges in the same order. The file is written through a buffer and flushed once at the end, and is listed in `.gitignore`. If `merges.bin` already exists, training is skipped and the file is loaded.
    - Reads `merges.bin` back with `loader()` and rebuilds the full ID-to-bytes vocabulary in memory (`build_vocab()`): 0-255 raw bytes, 256, 257, and every merged token as the bytes of its left part followed by its right part.
    - **Encodes text to IDs** (`encode()`): the text is split into chunks the same way as in training. `[cite]` and `<|endoftext|>` map straight to 256 and 257. Every other chunk starts as its bytes, then merges are applied repeatedly: of all adjacent pairs that have a learned merge, the one with the smallest merged ID (the one learned earliest) is applied everywhere in the chunk, and this repeats until no learned pair is left. Text never seen in training still encodes, because it falls back to raw bytes.
    - **Decodes IDs back to text** (`decode()`): looks up each ID in the vocabulary, joins the bytes, and converts them to a string. An unknown ID returns an error instead of panicking.
    - **Round-trip smoke test in `main()`:** encodes one sample sentence (containing `[cite]` and `<|endoftext|>`), prints the IDs, decodes them and asserts that the result equals the original.
  - **Test results so far** (code and full output in [docs/tokenizer_tests_results.md](docs/tokenizer_tests_results.md), all from the current `merges.bin`, trained on the current corpus):
    - Lossless round trip on the sample sentence, on edge cases (empty string, whitespace only, glued and malformed special tokens, code indentation, LaTeX and Unicode math, unseen strings), and on 10 whole papers.
    - Special-token counts match on every test string, and every ID stays below 16,384.
    - 10 papers: 487,146 bytes into 135,092 tokens, **3.61 bytes per token**. Caveats: those papers were part of the tokenizer's training data, so the figure is optimistic, the 10 files are not a random sample, and there is no baseline yet to say whether 3.61 is good for a 16k vocabulary.
    - **Determinism confirmed:** two fresh training runs (old `merges.bin` deleted each time) gave byte-identical files (`fc /b`). This was done on the earlier, smaller corpus (17.7 MB); it has not been repeated on the current one.
    - **Merge frequency curve** (frequency of the winning pair in the current training run):

      | Merge number | Frequency |
      |---|---|
      | 500 | 8,539 |
      | 2,000 | 1,286 |
      | 4,000 | 419 |
      | 8,000 | 110 |
      | 12,000 | 47 |
      | 14,000 | 34 |
      | 16,000 and 16,125 (last) | 25 |

      The last merge has frequency 25, so over 4 epochs the rarest tokens are seen about 100 times, which is thin for learning a good embedding. The final merges also tie at 25, so which pairs make it in is decided by the tie rule, not by a clear signal.
    - **What these tests do not prove:** a round trip passes even if the merges are wrong, because decoding only joins bytes. So they say nothing about whether the incremental pair counts are correct. The committed `main()` is the smoke test only; the larger test variants live in the results file.
  - **Not done yet:**
    - The incremental pair counts have not been checked against a full recount (see tests to add below).
    - Nothing encodes the whole dataset into a stored token stream yet; that belongs to the ingestion stage. Doing this also gives the real token count.
    - `encode()` rebuilds its merge lookup table on every call. This is fine for tests but must be built once and passed in before encoding a whole corpus.
    - `decode()` converts bytes to text leniently (invalid UTF-8 becomes a replacement character instead of an error). This is harmless for a complete sequence, but a cut-off sequence can end in the middle of a multi-byte character.
    - The input path is relative (`../dataset/arxiv_texts`) and the output goes to the current folder, so for now the program must be run from inside `tokenizer_core`.
    - The results file has a missing newline in the effectiveness output (two result lines are joined), which should be fixed.
  - **Tests to add (in this order):**
    1. Training with incremental updates must give exactly the same merges as a slow version that recounts every pair each iteration, on a small corpus (about 20 lines, 50 merges). Also compare the counts against a full recount every few merges. This is the most important open check.
    2. For each paper, the number of `[cite]` in the text must equal the number of ID 256 in its encoding (and likewise `<|endoftext|>` and 257).
    3. `merges.bin` size must be a multiple of 6 and equal 6 x the expected number of merges. `loader()` already asserts the multiple of 6; add the count check.
    4. Encoding text from the training data must give the segmentation that training produced (the replay order really matches the learned order).
    5. Measure characters per token on held-out papers, and compare with an existing tokenizer to get a baseline.
  - **Open decisions:**
    - **Vocabulary size: 16,384 is probably too large for this data.** The embedding table has (vocabulary size x model width) parameters, and the whole data-limited budget is about 2.5M. For illustration (model width not chosen yet): 16,384 tokens at width 128 is about 2.1M parameters, nearly the entire model; 4,000 tokens at width 128 is about 0.5M. At about 4,000 merges the winning frequency was 419, a healthy level. A range of roughly 4,000 to 8,000 looks more reasonable. Decide after the dataset is final: retrain, print the winning frequency at several points, check what fraction of the budget the embedding takes, and cut where frequency falls below roughly 50 to 100. Because training is deterministic and each merge depends only on the earlier ones, the first N records of `merges.bin` are exactly what an N-merge run would produce, so a smaller vocabulary can be tried by loading only the first N merges (the ID space then ends at 257 + N).
    - Whether the tokenizer should keep the metadata lines (`title--`, `arxiv id--`, ...) in the training text.
    - Whether a space should attach to the following word (currently spaces are separate chunks).

- ⏳ Dataset specialization matching: not started.
- ⏳ Rust ingestion pipeline (batching and streaming; tokenizing builds on `tokenizer_core`): not started.
- ⏳ Training loop (bf16 mixed precision): not started.
- ⏳ Inference with KV-cache: not started.

## Measuring the efficiency factor

The efficiency factor answers: "of the speed a pure matmul benchmark reaches, how much do I keep in real training?" Real training loses time to memory-bound operations (layer norm, softmax, optimizer step), small matrices, kernel-launch gaps, data loading, and laptop thermal throttling. Once the training loop exists, measure it instead of guessing:

1. Count the tokens per second the training loop processes.
2. Multiply by 6 x the parameter count to get the FLOPs per second actually achieved (about 8 x if gradient checkpointing is used).
3. Divide by the benchmark TFLOPS (measured in the same precision).
4. Run for long enough that thermal throttling shows up, then put the result into `sizer_core/src/main.rs`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` contains `torch` (the TFLOPS benchmark), `arxiv` and `requests` (finding and fetching arXiv papers), and `beautifulsoup4` (parsing the ar5iv HTML). The correct PyTorch build depends on your hardware (NVIDIA, AMD, or CPU-only), so use the selector on the PyTorch website if the default `pip install torch` does not match your machine.

### Building the Rust programs

To build the sizing program (requires [Rust](https://www.rust-lang.org/tools/install)):

```bash
cd sizer_core
cargo build --release
```

To build the tokenizer program (work in progress; it depends on the `regex` and `indexmap` crates, which Cargo downloads on first build):

```bash
cd tokenizer_core
cargo build --release
```

Rebuild with `--release` after any change to the Rust code, because `architecture.py` runs the release binary of `sizer_core`.

### Dataset folder

The `dataset/` folder is not part of the repository (it is listed in `.gitignore`), because it holds large text files and unchecked copyright data. It looks like this:

```
dataset/
  arxiv_texts/       created by download_arxiv_text.py
```

### Running

Run the Python scripts from the project root:

```bash
python download_arxiv_text.py     # downloads and cleans arXiv papers as text (needs internet)
python architecture.py            # hardware detection, benchmark, token count and sizing
```

The tokenizer is still not connected and is run on its own from inside its folder. It reads `../dataset/arxiv_texts` and trains, unless `merges.bin` already exists, in which case it skips training and loads the file. It then rebuilds the vocabulary and runs the round-trip sample test (encode, decode, compare). To retrain, delete `merges.bin` first (otherwise the run only loads the old file and tests nothing new), and do this whenever the dataset changes:

```bash
cd tokenizer_core
cargo run --release
```

## Requirements

- Python 3.x
- Rust toolchain (for `sizer_core` and `tokenizer_core`)
- For GPU detection: NVIDIA drivers (`nvidia-smi`) or ROCm tooling (`amd-smi` / `rocm-smi` / `rocminfo`), depending on hardware. CPU-only systems fall back automatically.
- For the intended training setup: a GPU with native bf16 support (NVIDIA Ampere or newer, such as the RTX 3050).
- Internet access for the download script

## Contributing

Contributions are welcome! Since the NVIDIA hardware paths have been successfully validated, community testing is now focused entirely on **AMD GPUs**.

If you have an AMD GPU, you can help immediately by running our architecture scripts and reporting how they handle your system topology.

Please check out our active tracking issue to help us validate:
👉 **[Issue #1: Help Needed: Live testing & validation of GPU detection paths (NVIDIA/AMD)](https://github.com/ojasgarg77/SLM-CREATOR/issues/1#issue-6224761352)**

For other bugs, documentation fixes, or dataset cleaning improvements, feel free to open a separate issue or submit a pull request.

## License

Copyright 2026 ojasgarg77

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://apache.org](http://apache.org) or in the [LICENSE](LICENSE) file in this repository.
