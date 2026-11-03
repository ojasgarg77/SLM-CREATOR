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
