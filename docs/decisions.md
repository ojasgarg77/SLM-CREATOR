# Decisions log

One entry per decision or dropped idea, with the reason. Code history lives in git; this file keeps the reasoning.

## 2026-10-25

- **arXiv source: PDFs -> ar5iv HTML.** `download_arxiv_data.py` (PDF download) replaced by `download_arxiv_text.py`. Reason: HTML exposes math as LaTeX in an annotation element, and citations, tables and authors can be identified by class.
- **Tables: drop by content, not by class name.** Dropping every `ltx_tabular` also deleted multi-line display equations in some papers (1901.06610: 8 of 12 equation rows sat inside a tabular). Rule now: keep a table only if it contains an equation row (`eqn_row` class). Tested on 2 papers so far (1901.06610, 2103.07492); needs more.
- **`ltx_tag`: keep section numbers, drop equation labels.** `ltx_tag` is used for section numbers, equation labels, captions, footnotes and bibliography items. Rule: drop a tag only if its parent's class contains `eqno`. Reason: the model should not learn to emit `(6)` after every formula.
- **Equation mentions ("Eq. 2") left as they are for now.** Labels are gone, so mentions point at nothing. Options considered: leave, placeholder token (like `[CITE]`), strip. Parked until there are many papers to count real spellings.
- **Hardware speed: own benchmark only.** An unsourced claim of 35-45 TFLOPS for the RTX 4050 was rejected; spec-sheet fp32 is about 9 TFLOPS (Notebookcheck), real value comes from the matmul benchmark on that laptop.
- **Fiction (BookCorpus): at most ~10% of the mix, or none.** About 10M of the 11.1M tokens counted so far; not on topic. Final mix to be decided with small experiments. *(Superseded 2026-11-21: the English data was dropped entirely.)*
- **Licensing: filter before any bulk download.** Excluded: arXiv's perpetual non-exclusive license. Usable: CC0, CC BY. Careful: BY-SA / NC variants. Skipped in v1: NC-ND.

## 2026-11-05

- **Replaced HashMap with IndexMap.** I chose `indexmap` over `HashMap` and accepted the extra dependency.
  - A `HashMap` cannot look up items by numeric position without a slow manual search; `IndexMap` can do it quickly.
  - `HashMap` iterates in a scrambled, random order (a defence against HashDoS attacks). I do not need that defence, and the random order is a real drawback for BPE: when two pairs have the same count, the winner would change from run to run. With `IndexMap` the order is the insertion order, so two runs on the same data give the same merges.
  - Update 2026-11-15: the tie rule is now explicit (see below), so determinism no longer depends on the map's order. `IndexMap` is kept for now; revisit whether the dependency is still worth it.

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
    - Costs I accept: the file cannot be read by eye (a small debug print or dump tool will be needed), and it has no header or version number. If the format changes (for example a vocabulary above 65,535, which would need `u32`, or a change to the special-token list), old files would be misread silently. Possible fix later: a few header bytes with a format version and the merge count. *(This cost became real on 2026-11-21, see below.)*
  - Check to add: file size must be a multiple of 6 and equal 6 x the number of merges.
- **`merges.bin` is not committed.** Added to `.gitignore`. It is generated output that can be rebuilt from the dataset, and the dataset itself is not in the repository.
- **End-of-document token `<|endoftext|>` = ID 257.** `read_files` appends it after every paper. *(Merged tokens originally started at 258; they now start at 264, see 2026-11-21.)*
  - Reason: without a separator the end of one paper runs straight into the start of the next, and the model would learn that unrelated text follows each other. At inference it also gives the model a way to say "this document is finished".
  - It is matched first in the regex and kept as one chunk, like `[cite]`, so merges can never split it or glue it to a neighbour.
  - Special tokens take the fixed IDs just above the raw bytes, before any learned merge, so their IDs never change when the number of merges changes.
- **Math text: strip LaTeX font commands and whitespace inside `$...$`.** `\mathrm`, `\mathbf`, `\it`, `\bf` and similar only change how a formula looks, not what it means, and each variant gives the tokenizer a different spelling of the same thing. Spaces inside math are removed for the same reason. *(Review on 2026-11-21 found three problems, fixed the same day, see below.)*

## 2026-11-13

- **Explicit BPE tie-breaking rule: smallest `(left, right)` token ID pair wins.** Replaced implicit `max_by_key` insertion-order tie resolution with an explicit `max_by` comparison: compare frequency first, and on a tie pick the lexicographically smallest `(left, right)` token ID tuple (`b.0.cmp(a.0)`).
  - Reason: relying on `IndexMap` insertion order means changing file read order or switching to an incremental pair-counter later could silently change which merges get learned. Comparing token IDs directly guarantees identical merges regardless of traversal order.
- **Vocabulary (`HashMap<u16, Vec<u8>>`) is rebuilt in RAM from `merges.bin`, not stored in a separate file.** `loader` reads the 6-byte records from `merges.bin`, and `build_vocab` reconstructs the byte sequences for the raw bytes, the special tokens and every merged ID in creation order.
  - Reason: `merges.bin` already encodes the full merge tree. Rebuilding the lookup table at startup takes under a millisecond and avoids keeping two files in sync or inventing a variable-length binary format for multi-byte tokens.
- **TFLOPS benchmark (`architecture.py`) hardened for laptops and multi-GPU systems.**
  - **12-second thermal soak + median reporting:** Replaced the 10-iteration warmup with a 12-second continuous matmul loop so laptop GPUs settle past short-term boost clocks into sustained TDP, and switched from mean to median TFLOPS to ignore background OS spikes.
  - **Native BF16 only (`including_emulation=False`):** Checks `torch.cuda.is_bf16_supported(including_emulation=False)` on PyTorch 2.4+ (falling back to compute capability `>= 8.0` on older versions) so pre-Ampere GPUs use native FP32 instead of slow software-emulated BF16.
  - **Vendor-to-PyTorch GPU index verification:** Sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before importing PyTorch and matches the vendor tool's reported GPU name against `torch.cuda.get_device_name()` (`resolve_pytorch_index`) so iGPU + dGPU or multi-GPU machines benchmark the selected card.

## 2026-11-15

- **Each distinct chunk is stored once, with a count.** The text is split into chunks, and the same chunk (for example the word "the" or a single space) repeats thousands of times. A `HashMap<Vec<u16>, usize>` counts every distinct chunk while reading, then it is turned into a `Vec<(chunk, count)>` that the merge loop can edit.
  - Reason: merging a word once and multiplying by its count gives exactly the same pair counts as merging every copy, with far less work and memory.
  - The order of the `Vec` is arbitrary (it comes out of a `HashMap`). That is fine because pair counts are sums and the tie rule is explicit, so the learned merges do not depend on the order.
- **Pair counts are computed once and updated incrementally, not recounted every iteration.** The old loop recounted every pair in every chunk for each merge, which cost (corpus size x number of merges). Now, after each merge:
  - the winning pair's entry is removed from the counts;
  - only the chunks that contain the winning pair are touched: their old pairs are subtracted (times the chunk count), the chunk is rewritten, and its new pairs are added (times the chunk count).
  - Risk: a mistake in this bookkeeping would silently give wrong counts and therefore different merges. *(Checked against a full recount on 2026-11-21, see below.)*
  - Side effect: `swap_remove` reorders the `IndexMap`, which is harmless now that the tie rule is explicit.
- **Vocabulary size set to 16,384 tokens (`NET_MERGES` = 16,126).** 256 bytes + 2 special tokens + 16,126 merges = 16,384 (2^14), chosen as a power of two for GPU alignment.
  - Caveat (added 2026-11-19): the real requirement for tensor-core efficiency is usually that dimensions are multiples of 8 (or 64 for the best kernels), not that they are powers of two. I should not claim that a power of two is required, and I have not measured any speed difference.
  - *(Superseded 2026-11-21: the vocabulary is now 9,216, see below.)*
- **PDF input removed.** `data_extractor.py` no longer reads PDFs; `pymupdf` and `dataset/pdfs/` are gone from the setup.
  - Reason: arXiv text now comes from the ar5iv HTML download (see 2026-10-25), which keeps math as LaTeX and identifies citations, tables and authors by class, so the PDF path was no longer needed.

## 2026-11-19

- **Encoding replays merges in the order they were learned: lowest merged ID first.** For each chunk, `encode` looks at all adjacent pairs, picks the one whose merge has the smallest new token ID (learned earliest), applies it everywhere in the chunk, and repeats until no learned pair is left.
  - Reason: during training the most frequent pair was merged first, then the next, so the tokens of any word were formed in ID order. Replaying in the same order reproduces the segmentation training would have produced. Applying merges in any other order (for example left to right) could give a different, worse split of the same word.
  - The same `split_text` function is used by training and by `encode`, so both cut the text into identical chunks. Merges never cross a chunk boundary in either.
  - Special tokens skip the merge step and map straight to their IDs.
  - Text never seen in training still encodes: whatever cannot be merged stays as raw bytes (IDs 0-255). This is the main reason for working on bytes instead of characters; there is no unknown token.
  - Known costs, accepted for now:
    - The merge lookup table must be built once and reused before encoding a whole corpus.
    - Each round scans every pair in the chunk and rewrites the chunk, so the cost per chunk grows quickly with its length. Chunks are single words or whitespace runs, so they are short, but a very long unbroken string would be slow.
    - Not yet verified that encoding text from the training data reproduces exactly the training segmentation.
- **Decoding concatenates the bytes of each token and converts to a string with `from_utf8_lossy`; an unknown ID returns an error.**
  - Reason: every token is a byte sequence, so decoding is just a lookup and a join. An unknown ID means a bug or a corrupt stream, so it is reported instead of skipped silently.
  - Caveat: `from_utf8_lossy` replaces invalid UTF-8 with a replacement character instead of failing. For a full encode-decode round trip this never happens, but a token boundary can fall in the middle of a multi-byte character, so decoding a partial sequence (for example text generated one token at a time) can show replacement characters temporarily.
- **A round-trip check runs at the start of the tokenizer's `main()`:** one sample sentence is encoded, decoded, and compared with the original using `assert_eq!`. It is a cheap guard against a broken `build_vocab`, `encode` or `decode`, and proves that special tokens survive. It is one sentence, not a test suite (larger tests: see 2026-11-21).

## 2026-11-21

- **Dropped `download_english_data` and `english_dataset`; ingestion is arXiv-only.** Reason: aligns the setup with the domain-specific arXiv corpus rather than off-topic fiction.
- **Structural tags added as special tokens; merges now start at 264.** The downloader writes `[title]`, `[abstract]`, `[body]`, `[eq]` (display equations), `[proof]` and `[section]` (top-level section titles) in the text, and `[cite]` and `<|endoftext|>` as before. Each is one fixed token: `[cite]` 256, `<|endoftext|>` 257, `[title]` 258, `[abstract]` 259, `[body]` 260, `[section]` 261, `[eq]` 262, `[proof]` 263.
  - Reason : the model sees document structure as single, unmergeable tokens, so it can learn where equations, proofs and sections start. The tags are matched first in the regex, like `[cite]`, so merges cannot split them or glue them to neighbours.Some of these also help later on in infrencing.
  - The `arxiv id--` line was dropped; title and abstract stay, lowercased and whitespace-normalized, with the tags as markers. Sub-section titles are not tagged, and a proof block is flattened into one block (no `[eq]` inside it).
  - Because the downloader skips existing files, the whole corpus had to be re-downloaded in the new format and the tokenizer retrained. Done.
  - Known risk, accepted for now: the tag list is written separately in `split_text` (two regexes), `bpe`, `build_vocab`, `encode`, the tests and the toy corpus, plus a hardcoded 264 and some stale comments. A missed place gives silent mismatches (for example training that splits a tag into bytes while encoding maps it to one ID). Planned fix: one shared table with text and ID together, from which the regexes, the lookup and the first free ID are derived.
  - A second consequence: an old `merges.bin` (trained before the tags) would load without error and overwrite the tag entries in `build_vocab`. Planned guard in `loader`: first merged ID equals the first free ID (256 + number of specials), IDs are consecutive, both children are smaller than the new ID, and the merge count is the expected one. Not implemented yet.
- **Vocabulary reduced to 9,216 (`NET_MERGES` = 8,952).** 256 bytes + 8 special tokens + 8,952 merges = 9,216 (a multiple of 64).
  - Reason: on the first corpus the 16,384-token run ended with a last-merge frequency of 10 (25 on the second corpus), and the embedding table (vocabulary x model width) would take most of a roughly 2.5M-parameter budget. For illustration: 16,384 x 128 is about 2.1M.
  - Current run: winning-pair frequency was 8,372 at merge 500, 3,386 at 1,000, 1,273 at 2,000, 409 at 4,000, 106 at 8,000 and 84 at the last merge (iteration 8,951). About 84 x 4 epochs is roughly 330 sightings for the rarest tokens, a reasonable level.
  - Still open: 9,216 x 128 is about 1.2M (nearly half of the budget), 9,216 x 64 about 0.6M. Revisit once the model width is chosen.
  - Because training is deterministic and each merge depends only on earlier ones, the first N records of `merges.bin` are exactly what an N-merge run would produce, so smaller vocabularies can be compared by loading only the first N merges (the ID space then ends at 263 + N). Smaller vocabularies compress less, so the token count changes with the choice and must be re-measured.
- **Tokenizer tests; results in `docs/tokenizer_tests_results.md`, now one unified block labeled with `NET_MERGES`.** Checked: lossless round trip on a sample with all specials, on edge cases (adjacent and malformed tags, empty and whitespace-only strings, nested-brace math, Unicode) and on 10 papers; special-token counts per string and per paper; ID range; and the incremental recount below. Result on 10 papers: 489,897 bytes into 139,784 tokens, 3.50 bytes per token.
  - Limits: a round trip passes even with wrong merges. The 10 papers were in the tokenizer's training data, are not a random sample, and there is no baseline, so 3.50 is optimistic. It is also not a controlled comparison with the earlier 3.61 (different vocabulary and files). Per-paper tag counts match trivially on an old-format file (zero tags, zero IDs), which is why the stale-file check was needed.
  - Hygiene rule: every result block must say which `merges.bin` (here, `NET_MERGES`) it came from. Earlier the file mixed results from two different `merges.bin` files.
- **Incremental pair counts verified against a full recount.** A test (`verify_incremental_bpe`) runs the incremental update and a naive recount side by side on a toy corpus and compares the winning pair and its count at every step; it passed across 40 merges.
  - Limits: it tests a copy of the merge loop, not `bpe()` itself; it shares `apply_merge` and `count_pairs` with the real path; it compares only the winner (a wrong subtraction for an absent pair is silently ignored by `adjust_pairs`); and the toy corpus has no long chunks or multi-byte text. Planned: call the same merge-step function from both, compare the full maps every few steps, count the steps actually run, and run it once on a real paper sample.
- **Determinism check: two fresh training runs gave byte-identical `merges.bin`.** Compared with `fc /b` after deleting the old file before each run, on the earlier corpus and tokenizer version. Rust's `HashMap` order is randomized per run, so this shows the explicit tie rule and sum-based counts remove any dependence on visiting order. It does not show that the merges are correct. To repeat on the current setup.
- **Token count estimate: characters / 3.5, as an integer.** `data_extractor.py` estimates 12,714,236 tokens (about 44.5M characters), so the data limit (tokens x 4 / 20) is about 2.5M parameters. Data is the binding constraint: every other limit is far higher.
  - 3.5 comes from the 10-paper check above. It is a bytes per token figure, while the extractor counts characters without newlines; the text is mostly ASCII, so the difference is probably small. Counting file bytes would match the units. The ratio must be recomputed whenever the vocabulary or data changes.
  - Bug fixed earlier the same day: dividing with `//` by a float gives a float (`12286188.0`), and the Rust sizer parses the token count as an integer and would exit with an error. The value is now converted with `int(...)`.
  - Correction to an earlier claim: the old guess of 4 characters per token under-counted tokens (smaller data limit); it did not overestimate them.
- **Plan the data collection from the target size.** Working backwards from the sizer's rule (tokens = 5 x parameters): about 100M tokens (roughly 350 MB of text) supports about 20M parameters, so the corpus is about 8 times too small for that. A 24-hour run at the current benchmark and the 0.3 placeholder factor supports roughly 44M parameters (hand calculation), so data beyond that has no use yet.
  - Order: scale up the existing arXiv pipeline first (it already handles math, citations and cleaning); run a batch of a few hundred papers to measure average cleaned size and failure rate; only add a second source type (textbooks, lecture notes) if arXiv runs short, and keep off-topic text small. Open: the licensing question (see 2026-10-25) must be answered before a bulk download.
- **Downloader reruns are cheap and "N" means new papers (implemented).** The search is iterated lazily (no `list(...)`), the user's number is "N new papers" (only successful saves count; skipped and failed papers do not), and the search `max_results` is a separate safety cap of 5,000. The `stat.ML` term now has its `cat:` prefix. The minimum cleaned length was raised from 100 to 500 characters.
  - Known remaining costs: each run still walks past existing papers from the top of the results (about one page of 100 per 100 existing papers); the 500-character minimum will not catch partial conversions like the Adam paper (1412.6980); there is no end-of-run summary and no stop after repeated network errors.
- **Math cleaning: three review findings fixed, a few left open.**
  - Fixed: `\beta` is no longer in the font-command list (it is a real symbol); `strip_font_commands` removes the command, its opening brace and only the matching closing brace by counting braces, so nested braces stay balanced, and an unclosed command no longer loops forever; `clean_math_whitespaces` keeps one space after a backslash-command followed by a letter, so `\sum i` does not become `\sumi`.
  - Still open: `\text{...}` and `\operatorname{...}` lose their spaces, escaped braces `\{` `\}` are counted as real braces, the abstract and title are lowercased with raw LaTeX left uncleaned (lowercasing turns `\Delta` into `\delta`), and these functions have no tests of their own (suggested inputs: `\mathbf{W_{1}}`, `\mathrm{d}^{2}`, `\mathbf{\mathrm{x}}`, `\beta{x}`, `\sum i`, an unclosed `\mathbf{W`, with balanced braces checked).
