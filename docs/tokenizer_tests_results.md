# Tokenizer Tests & Results

## 1. Effectiveness Check Variant (`fn main`)

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

    let tokenizer = Tokenizer::new(&merges);

    println!("STARTING SAMPLE TESTS\n");
    let sample = "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>";
    let ids = tokenizer.encode(sample);
    println!("SAMPLE: {:?}", sample);
    println!("TOKEN IDS({} tokens):{:?}", ids.len(), ids);

    let decoded = tokenizer.decode(&ids).unwrap();
    println!("DECODED: {:?}", decoded);
    assert_eq!(sample, decoded);

    let test_cases = [
        "",
        "   \n\t  ",
        "backpropagation gradient descent adam optimizer neural network",
        "loss = cross_entropy(logits, targets) where $x_i \\in \\mathbb{R}^d$ [cite]",
        "unseen_xyz_token_12345 and normal words together [cite] <|endoftext|>",
        "π θ α β λ — “quotes” … spaces   test",
    ];

    for case_str in test_cases {
        let case_ids = tokenizer.encode(case_str);
        let case_decoded = tokenizer.decode(&case_ids).unwrap();
        assert_eq!(case_str, case_decoded, "ROUND-TRIP MISMATCH ON: {:?}", case_str);

        let bytes_len = case_str.as_bytes().len();
        let chars_len = case_str.chars().count();
        let tok_len = case_ids.len().max(1);
        let bytes_per_tok = bytes_len as f64 / tok_len as f64;
        let chars_per_tok = chars_len as f64 / tok_len as f64;
        let cite_count = case_ids.iter().filter(|&&id| id == 256).count();
        let eot_count = case_ids.iter().filter(|&&id| id == 257).count();

        println!(
            "toks={:2} | bytes={:3} | B/T={:.2} | C/T={:.2} | [cite]={}/{} | eot={}/{} | {:?}",
            case_ids.len(),
            bytes_len,
            bytes_per_tok,
            chars_per_tok,
            cite_count,
            case_str.matches("[cite]").count(),
            eot_count,
            case_str.matches("<|endoftext|>").count(),
            case_str
        );
    }

    if let Ok(mut entries) = fs::read_dir("../dataset/arxiv_texts") {
        let mut total_bytes = 0usize;
        let mut total_tokens = 0usize;
        let mut files_checked = 0usize;

        while let Some(Ok(entry)) = entries.next() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
                if let Ok(text) = fs::read_to_string(&path) {
                    let paper_ids = tokenizer.encode(&text);
                    assert_eq!(text, tokenizer.decode(&paper_ids).unwrap(), "CORPUS ROUND-TRIP FAIL: {:?}", path);
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
```

### Effectiveness Check Results

```text
FOUND EXISTING merges.bin. SKIPPING BPE TRAINING.

LOADED 16126 MERGE RULES.
STARTING SAMPLE TESTS

SAMPLE: "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>"
TOKEN IDS(13 tokens):[78, 101, 1381, 685, 2208, 1326, 32, 256, 1718, 32, 256, 32, 257]
DECODED: "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>"
toks= 0 | bytes=  0 | B/T=0.00 | C/T=0.00 | [cite]=0/0 | eot=0/0 | ""
toks= 5 | bytes=  7 | B/T=1.40 | C/T=1.40 | [cite]=0/0 | eot=0/0 | "   \n\t  "
toks= 7 | bytes= 62 | B/T=8.86 | C/T=8.86 | [cite]=0/0 | eot=0/0 | "backpropagation gradient descent adam optimizer neural network"
toks=30 | bytes= 73 | B/T=2.43 | C/T=2.43 | [cite]=1/1 | eot=0/0 | "loss = cross_entropy(logits, targets) where $x_i \\in \\mathbb{R}^d$ [cite]"
toks=18 | bytes= 69 | B/T=3.83 | C/T=3.83 | [cite]=1/1 | eot=1/1 | "unseen_xyz_token_12345 and normal words together [cite] <|endoftext|>"toks=31 | bytes= 49 | B/T=1.58 | C/T=1.16 | [cite]=0/0 | eot=0/0 | "π θ α β λ — “quotes” … spaces   test"

FIRST 10 PAPERS: 487146 bytes -> 135092 tokens (3.61 bytes/token)
```

---

## 2. Baseline Sample Test Variant (`fn main`)

```rust
fn main(){
    if !Path::new("merges.bin").exists(){
        println!("STARTING BPE....\n\n");
        let file_contents=read_files();
        bpe(&file_contents);
    }else{
        println!("FOUND EXISTING merges.bin. SKIPPING BPE TRAINING.\n");
    }

    let merges = loader("merges.bin");
    println!("LOADED {} MERGE RULES.", merges.len());

    let tokenizer = Tokenizer::new(&merges);


    println!("STARTING SAMPLE TESTS\n");
    let sample="Neural networks learned through [cite] backpropagation [cite] <|endoftext|>";
    let ids=tokenizer.encode(sample);
    println!("SAMPLE: {:?}",sample);
    println!("TOKEN IDS({} tokens):{:?}",ids.len(),ids);

    let decoded=tokenizer.decode(&ids).unwrap();
    println!("DECODED: {:?}",decoded);
    assert_eq!(sample, decoded);
}
```

---

## 3. Niche Edge-Case Check Variant (`fn main`)

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

    let tokenizer = Tokenizer::new(&merges);

    println!("STARTING SAMPLE TESTS\n");
    let sample = "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>";
    let ids = tokenizer.encode(sample);
    println!("SAMPLE: {:?}", sample);
    println!("TOKEN IDS({} tokens):{:?}", ids.len(), ids);

    let decoded = tokenizer.decode(&ids).unwrap();
    println!("DECODED: {:?}", decoded);
    assert_eq!(sample, decoded);

    let niche_cases = [
        "[cite][cite]<|endoftext|><|endoftext|>[cite]",
        "cite] [cite <endoftext> <|endoftext my_cite_function",
        "(see [cite],eq. 3)<|endoftext|>Next paper",
        "h^{(l+1)}_t = RMSNorm(x_t + Attn(RoPE(q_t), RoPE(k_t), v_t))",
        "SwiGLU(x, W, V, W_2) = (SiLU(xW) * (xV))W_2",
        "∇_θ L(θ) ≈ ∑_{i=1}^n ∂ℓ_i/∂θ  ‖x‖₂ ≤ 1  ﬁ ﬂ",
        "\n\n\n    def forward(self, x):\n        return x",
        " [cite]  <|endoftext|> ",
        "token token token token token [cite] [cite] [cite]",
    ];

    for case_str in niche_cases {
        let case_ids = tokenizer.encode(case_str);
        assert!(
            case_ids.iter().all(|&id| (id as usize) < 16384),
            "OUT-OF-VOCAB ID IN: {:?}",
            case_str
        );

        let case_decoded = tokenizer.decode(&case_ids).unwrap();
        assert_eq!(case_str, case_decoded, "ROUND-TRIP MISMATCH ON: {:?}", case_str);

        let bytes_len = case_str.as_bytes().len();
        let chars_len = case_str.chars().count();
        let tok_len = case_ids.len().max(1);
        let bytes_per_tok = bytes_len as f64 / tok_len as f64;
        let chars_per_tok = chars_len as f64 / tok_len as f64;
        let cite_count = case_ids.iter().filter(|&&id| id == 256).count();
        let eot_count = case_ids.iter().filter(|&&id| id == 257).count();

        println!(
            "toks={:2} | bytes={:3} | B/T={:.2} | C/T={:.2} | [cite]={}/{} | eot={}/{} | {:?}",
            case_ids.len(),
            bytes_len,
            bytes_per_tok,
            chars_per_tok,
            cite_count,
            case_str.matches("[cite]").count(),
            eot_count,
            case_str.matches("<|endoftext|>").count(),
            case_str
        );
    }

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
                        paper_ids.iter().all(|&id| (id as usize) < 16384),
                        "OUT-OF-VOCAB ID IN PAPER: {:?}",
                        path
                    );
                    assert_eq!(
                        text,
                        tokenizer.decode(&paper_ids).unwrap(),
                        "CORPUS ROUND-TRIP FAIL: {:?}",
                        path
                    );
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
```

### Niche Edge-Case Results

```text
FOUND EXISTING merges.bin. SKIPPING BPE TRAINING.

LOADED 16126 MERGE RULES.
STARTING SAMPLE TESTS

SAMPLE: "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>"
TOKEN IDS(13 tokens):[78, 101, 1381, 685, 2208, 1326, 32, 256, 1718, 32, 256, 32, 257]
DECODED: "Neural networks learned through [cite] backpropagation [cite] <|endoftext|>"
toks= 5 | bytes= 44 | B/T=8.80 | C/T=8.80 | [cite]=3/3 | eot=2/2 | "[cite][cite]<|endoftext|><|endoftext|>[cite]"
toks=24 | bytes= 52 | B/T=2.17 | C/T=2.17 | [cite]=0/0 | eot=0/0 | "cite] [cite <endoftext> <|endoftext my_cite_function"
toks=13 | bytes= 41 | B/T=3.15 | C/T=3.15 | [cite]=1/1 | eot=1/1 | "(see [cite],eq. 3)<|endoftext|>Next paper"
toks=44 | bytes= 60 | B/T=1.36 | C/T=1.36 | [cite]=0/0 | eot=0/0 | "h^{(l+1)}_t = RMSNorm(x_t + Attn(RoPE(q_t), RoPE(k_t), v_t))"
toks=40 | bytes= 43 | B/T=1.07 | C/T=1.07 | [cite]=0/0 | eot=0/0 | "SwiGLU(x, W, V, W_2) = (SiLU(xW) * (xV))W_2"
toks=59 | bytes= 70 | B/T=1.19 | C/T=0.73 | [cite]=0/0 | eot=0/0 | "∇_θ L(θ) ≈ ∑_{i=1}^n ∂ℓ_i/∂θ  ‖x‖₂ ≤ 1  ﬁ ﬂ"
toks=14 | bytes= 45 | B/T=3.21 | C/T=3.21 | [cite]=0/0 | eot=0/0 | "\n\n\n    def forward(self, x):\n        return x"
toks= 5 | bytes= 23 | B/T=4.60 | C/T=4.60 | [cite]=1/1 | eot=1/1 | " [cite]  <|endoftext|> "
toks=11 | bytes= 50 | B/T=4.55 | C/T=4.55 | [cite]=3/3 | eot=0/0 | "token token token token token [cite] [cite] [cite]"

FIRST 10 PAPERS: 487146 bytes -> 135092 tokens (3.61 bytes/token)
```