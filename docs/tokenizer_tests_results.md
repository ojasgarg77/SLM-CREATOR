# Tokenizer Tests & Results

## 1. Unified Effectiveness, Structural Tag, Edge-Case & Incremental Verification (`fn main`)

```rust
fn main() {
    if !Path::new("merges.bin").exists() {
        println!("STARTING BPE....\n\n");
        let file_contents = read_files();
        bpe(&file_contents);
    } else {
        println!("FOUND EXISTING merges.bin. SKIPPING BPE TRAINING.\n");
    }

    let merges = loader("merges.bin");
    println!("LOADED {} MERGE RULES.", merges.len());
    let max_vocab_id = 264 + merges.len();

    let tokenizer = Tokenizer::new(&merges);

    println!("STARTING SAMPLE & STRUCTURAL TAG TESTS\n");
    let sample = "[title] attention [abstract] neural networks [cite] [body] [section] intro [eq] $x=y$ [proof] done <|endoftext|>";
    let ids = tokenizer.encode(sample);
    println!("SAMPLE: {:?}", sample);
    println!("TOKEN IDS ({} tokens): {:?}", ids.len(), ids);

    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(sample, decoded);
    for expected_id in 256..=263u16 {
        assert!(
            ids.contains(&expected_id),
            "MISSING SPECIAL TOKEN ID {} IN SAMPLE",
            expected_id
        );
    }

    let niche_cases = [
        "[cite][title][abstract][body][section][eq][proof]<|endoftext|>",
        "title] [title section] [eq <proof> <|endoftext my_cite_function",
        "[eq] $\\mathbf{W_{1}} + \\beta{x} = \\sum i$ [cite]",
        "[title] transformer\n[abstract] scaling laws\n[body] [section] method\n\n[eq] $y = Wx$\n\n[proof] let $x \\in \\mathbb{R}^d$.<|endoftext|>",
        "∇_θ L(θ) ≈ ∑_{i=1}^n ∂ℓ_i/∂θ  ‖x‖₂ ≤ 1  ﬁ ﬂ",
        "",
        "   \n\t  ",
    ];

    let special_tags: [(&str, u16); 8] = [
        ("[cite]", 256),
        ("<|endoftext|>", 257),
        ("[title]", 258),
        ("[abstract]", 259),
        ("[body]", 260),
        ("[section]", 261),
        ("[eq]", 262),
        ("[proof]", 263),
    ];

    for case_str in niche_cases {
        let case_ids = tokenizer.encode(case_str);
        assert!(
            case_ids.iter().all(|&id| (id as usize) < max_vocab_id),
            "OUT-OF-VOCAB ID IN: {:?}",
            case_str
        );

        let case_decoded = tokenizer.decode(&case_ids).unwrap();
        assert_eq!(case_str, case_decoded, "ROUND-TRIP MISMATCH ON: {:?}", case_str);

        for (tag_str, tag_id) in special_tags {
            let actual = case_ids.iter().filter(|&&id| id == tag_id).count();
            let expected = case_str.matches(tag_str).count();
            assert_eq!(
                actual, expected,
                "TAG COUNT MISMATCH FOR {} ON: {:?}",
                tag_str, case_str
            );
        }

        let bytes_len = case_str.as_bytes().len();
        let tok_len = case_ids.len().max(1);
        println!(
            "PASS | toks={:2} | bytes={:3} | B/T={:.2} | {:?}",
            case_ids.len(),
            bytes_len,
            bytes_len as f64 / tok_len as f64,
            case_str
        );
    }

    println!("\nVERIFYING INCREMENTAL PAIR COUNTS VS FULL RECOUNT...");
    verify_incremental_bpe();

    if let Ok(mut entries) = fs::read_dir("../dataset/arxiv_texts") {
        let mut total_bytes = 0usize;
        let mut total_tokens = 0usize;
        let mut files_checked = 0usize;

        while let Some(Ok(entry)) = entries.next() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(text) = fs::read_to_string(&path) {
                    let paper_ids = tokenizer.encode(&text);
                    assert!(
                        paper_ids.iter().all(|&id| (id as usize) < max_vocab_id),
                        "OUT-OF-VOCAB ID IN PAPER: {:?}",
                        path
                    );
                    assert_eq!(
                        text,
                        tokenizer.decode(&paper_ids).unwrap(),
                        "CORPUS ROUND-TRIP FAIL: {:?}",
                        path
                    );
                    for (tag_str, tag_id) in special_tags {
                        assert_eq!(
                            paper_ids.iter().filter(|&&id| id == tag_id).count(),
                            text.matches(tag_str).count(),
                            "PAPER TAG MISMATCH ({}) IN: {:?}",
                            tag_str,
                            path
                        );
                    }
                    total_bytes += text.as_bytes().len();
                    total_tokens += paper_ids.len();
                    files_checked += 1;
                    if files_checked >= 10 {
                        break;
                    }
                }
            }
        }

        if total_tokens > 0 {
            println!(
                "\nFIRST {} PAPERS: {} bytes -> {} tokens ({:.2} bytes/token)",
                files_checked,
                total_bytes,
                total_tokens,
                total_bytes as f64 / total_tokens as f64
            );
        }
    }
}

fn verify_incremental_bpe() {
    let toy_corpus = "[title] neural networks\n[abstract] aaaa bbbb aaaa gradient descent [cite]\n[body] [section] intro\n[eq] $x = aaaa + bbbb$\n[proof] aaaa bbbb backpropagation [cite] <|endoftext|>";
    let clean_text = split_text(toy_corpus);

    let mut chunk_counts: HashMap<Vec<u16>, usize> = HashMap::new();
    for chunk in &clean_text {
        let tokens: Vec<u16> = match *chunk {
            "[cite]" => vec![256],
            "<|endoftext|>" => vec![257],
            "[title]" => vec![258],
            "[abstract]" => vec![259],
            "[body]" => vec![260],
            "[section]" => vec![261],
            "[eq]" => vec![262],
            "[proof]" => vec![263],
            _ => chunk.as_bytes().iter().map(|&b| b as u16).collect(),
        };
        *chunk_counts.entry(tokens).or_insert(0) += 1;
    }

    let mut inc_chunks: Vec<(Vec<u16>, usize)> = chunk_counts.into_iter().collect();
    let mut naive_chunks = inc_chunks.clone();
    let mut pair_counts = count_pairs(&inc_chunks);
    let mut next_id: u16 = 264;

    for step in 0..40 {
        let inc_best = pair_counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(&p, &f)| (p, f));

        let naive_counts = count_pairs(&naive_chunks);
        let naive_best = naive_counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(&p, &f)| (p, f));

        assert_eq!(
            inc_best, naive_best,
            "INCREMENTAL VS NAIVE MISMATCH AT STEP {}",
            step
        );

        let Some((best_pair, _)) = inc_best else {
            break;
        };

        pair_counts.swap_remove(&best_pair);
        for (chunk, freq) in &mut inc_chunks {
            if !chunk.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1) {
                continue;
            }
            for pair in chunk.windows(2) {
                let p = (pair[0], pair[1]);
                if p != best_pair {
                    adjust_pairs(&mut pair_counts, p, -(*freq as isize));
                }
            }
            let merged = apply_merge(chunk, best_pair, next_id);
            for pair in merged.windows(2) {
                adjust_pairs(&mut pair_counts, (pair[0], pair[1]), *freq as isize);
            }
            *chunk = merged;
        }

        for (chunk, _) in &mut naive_chunks {
            if chunk.windows(2).any(|w| w[0] == best_pair.0 && w[1] == best_pair.1) {
                *chunk = apply_merge(chunk, best_pair, next_id);
            }
        }

        next_id += 1;
    }
    println!("PASS | INCREMENTAL PAIR UPDATES MATCH FULL RECOUNT ACROSS 40 MERGES.");
}
```

### Results (`NET_MERGES = 8952`, `vocab_size = 9216`)

```text
FOUND EXISTING merges.bin. SKIPPING BPE TRAINING.

LOADED 8952 MERGE RULES.
STARTING SAMPLE & STRUCTURAL TAG TESTS

SAMPLE: "[title] attention [abstract] neural networks [cite] [body] [section] intro [eq] $x=y$ [proof] done <|endoftext|>"
TOKEN IDS (27 tokens): [258, 1080, 32, 259, 609, 682, 32, 256, 32, 260, 32, 261, 921, 306, 32, 262, 32, 36, 120, 61, 121, 36, 32, 263, 3467, 32, 257]
PASS | toks= 8 | bytes= 62 | B/T=7.75 | "[cite][title][abstract][body][section][eq][proof]<|endoftext|>"
PASS | toks=29 | bytes= 63 | B/T=2.17 | "title] [title section] [eq <proof> <|endoftext my_cite_function"
PASS | toks=27 | bytes= 48 | B/T=1.78 | "[eq] $\\mathbf{W_{1}} + \\beta{x} = \\sum i$ [cite]"
PASS | toks=40 | bytes=130 | B/T=3.25 | "[title] transformer\n[abstract] scaling laws\n[body] [section] method\n\n[eq] $y = Wx$\n\n[proof] let $x \\in \\mathbb{R}^d$.<|endoftext|>"
PASS | toks=58 | bytes= 70 | B/T=1.21 | "∇_θ L(θ) ≈ ∑_{i=1}^n ∂ℓ_i/∂θ  ‖x‖₂ ≤ 1  ﬁ ﬂ"
PASS | toks= 0 | bytes=  0 | B/T=0.00 | ""
PASS | toks= 5 | bytes=  7 | B/T=1.40 | "   \n\t  "

VERIFYING INCREMENTAL PAIR COUNTS VS FULL RECOUNT...
PASS | INCREMENTAL PAIR UPDATES MATCH FULL RECOUNT ACROSS 40 MERGES.

FIRST 10 PAPERS: 489897 bytes -> 139784 tokens (3.50 bytes/token)
```

---