# Decisions log

One entry per decision or dropped idea, with the reason. Code history lives in git; this file keeps the reasoning.

## 2026-10-25

- **arXiv source: PDFs -> ar5iv HTML.** `download_arxiv_data.py` (PDF download) replaced by `download_arxiv_text.py`. Reason: HTML exposes math as LaTeX in an annotation element, and citations, tables and authors can be identified by class.
- **Tables: drop by content, not by class name.** Dropping every `ltx_tabular` also deleted multi-line display equations in some papers (1901.06610: 8 of 12 equation rows sat inside a tabular). Rule now: keep a table only if it contains an equation row (`eqn_row` class). Tested on 2 papers so far (1901.06610, 2103.07492); needs more.
- **`ltx_tag`: keep section numbers, drop equation labels.** `ltx_tag` is used for section numbers, equation labels, captions, footnotes and bibliography items. Rule: drop a tag only if its parent's class contains `eqno`. Reason: the model should not learn to emit `(6)` after every formula.
- **Equation mentions ("Eq. 2") left as they are for now.** Labels are gone, so mentions point at nothing. Options considered: leave, placeholder token (like `[CITE]`), strip. Parked until there are many papers to count real spellings.
- **Hardware speed: own benchmark only.** An unsourced claim of 35-45 TFLOPS for the RTX 4050 was rejected; spec-sheet fp32 is about 9 TFLOPS (Notebookcheck), real value comes from the matmul benchmark on that laptop.
- **Fiction (BookCorpus): at most ~10% of the mix, or none.** About 10M of the 11.1M tokens counted so far; not on topic. Final mix to be decided with small experiments.
- **Licensing: filter before any bulk download.** Excluded: arXiv's perpetual non-exclusive license. Usable: CC0, CC BY. Careful: BY-SA / NC variants. Skipped in v1: NC-ND.

## 2026-11-05

- **Replaced HashMap with IndexMap.** I chose `indexmap` over `HashMap` and accepted the extra dependency.
  - A `HashMap` cannot look up items by numeric position without a slow manual search; `IndexMap` can do it quickly.
  - `HashMap` iterates in a scrambled, random order (a defence against HashDoS attacks). I do not need that defence, and the random order is a real drawback for BPE: when two pairs have the same count, the winner would change from run to run. With `IndexMap` the order is the insertion order, so two runs on the same data give the same merges.

## 2026-11-09

- **Learned merges are stored in a binary file (`merges.bin`), not JSON.**
  - Format: one fixed 6-byte record per merge: left token, right token, new token ID, each a little-endian `u16`. Records are in the order the merges were learned, because encoding has to replay them in that order.
  - Why binary:
    - Every field is a `u16`, so a record is exactly 6 bytes. Reading is `from_le_bytes` on 6-byte slices; there is no text parsing, no quoting and no number-to-text conversion in either direction.
    - It is compact: 6 bytes per merge, against 20 or more as JSON text. A full vocabulary of about 65k merges is roughly 390 KB.
    - The file is the data, with no formatting choices (spacing, key order, number style) that could differ between runs or tools.
  - Why not JSON: it costs a parser, and numbers become text and back again. Its one real advantage is that I can open it and read it.
  - Honest caveats:
    - The speed gain is real but small. At 200 merges (1.2 KB) loading is instant either way; it only starts to matter near the full vocabulary. I picked binary mainly for simplicity of reading and writing fixed-size records, not because the current file is slow.
    - Binary is not "the industry standard" for tokenizers. GPT-2 uses a JSON vocabulary plus a plain-text merges file, tiktoken uses text files, and Hugging Face tokenizers use JSON. Binary is a legitimate choice, but I should not justify it by saying everyone does it.
    - Costs I accept: the file cannot be read by eye (a small debug print or dump tool will be needed), and it has no header or version number. If the format changes (for example a vocabulary above 65,535, which would need `u32`), old files would be misread silently. Possible fix later: a few header bytes with a format version and the merge count.
  - Check to add: file size must be a multiple of 6 and equal 6 x the number of merges.
- **`merges.bin` is not committed.** Added to `.gitignore`. It is generated output that can be rebuilt from the dataset, and the dataset itself is not in the repository.
- **End-of-document token `<|endoftext|>` = ID 257; merged tokens start at 258.** `read_files` appends it after every paper.
  - Reason: without a separator the end of one paper runs straight into the start of the next, and the model would learn that unrelated text follows each other. At inference it also gives the model a way to say "this document is finished".
  - It is matched first in the regex and kept as one chunk, like `[cite]`, so merges can never split it or glue it to a neighbour.
  - Special tokens take the fixed IDs just above the raw bytes (256, 257), before any learned merge, so their IDs never change when the number of merges changes.
- **Math text: strip LaTeX font commands and whitespace inside `$...$`.** `\mathrm`, `\mathbf`, `\it`, `\bf` and similar only change how a formula looks, not what it means, and each variant gives the tokenizer a different spelling of the same thing. Spaces inside math are removed for the same reason. To check on many papers: that nothing meaningful is lost (the `\beta` entry in the regex looks wrong and must be reviewed).


## 2026-11-13

- **Explicit BPE tie-breaking rule: smallest `(left, right)` token ID pair wins.** Replaced implicit `max_by_key` insertion-order tie resolution with an explicit `max_by` comparison: compare frequency first, and on a tie pick the lexicographically smallest `(left, right)` token ID tuple (`b.0.cmp(a.0)`).
  - Reason: relying on `IndexMap` insertion order means changing file read order or switching to an incremental pair-counter later could silently change which merges get learned. Comparing token IDs directly guarantees identical merges regardless of traversal order.
- **Vocabulary (`HashMap<u16, Vec<u8>>`) is rebuilt in RAM from `merges.bin`, not stored in a separate file.** `loader` reads the 6-byte records from `merges.bin`, and `build_vocab` reconstructs the byte sequences for `0..=255`, `256` (`[cite]`), `257` (`<|endoftext|>`), and every merged ID in creation order.
  - Reason: `merges.bin` already encodes the full merge tree. Rebuilding the lookup table at startup takes under a millisecond and avoids keeping two files in sync or inventing a variable-length binary format for multi-byte tokens.
- **TFLOPS benchmark (`architecture.py`) hardened for laptops and multi-GPU systems.**
  - **12-second thermal soak + median reporting:** Replaced the 10-iteration warmup with a 12-second continuous matmul loop so laptop GPUs settle past short-term boost clocks into sustained TDP, and switched from mean to median TFLOPS to ignore background OS spikes.
  - **Native BF16 only (`including_emulation=False`):** Checks `torch.cuda.is_bf16_supported(including_emulation=False)` on PyTorch 2.4+ (falling back to compute capability `>= 8.0` on older versions) so pre-Ampere GPUs use native FP32 instead of slow software-emulated BF16.
  - **Vendor-to-PyTorch GPU index verification:** Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before importing PyTorch and matches the vendor tool's reported GPU name against `torch.cuda.get_device_name()` (`resolve_pytorch_index`) so iGPU + dGPU or multi-GPU machines benchmark the selected card.
