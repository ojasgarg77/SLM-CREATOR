use std::{collections::HashMap, fs::{self, File}, io::{BufWriter,Write}, path::Path, sync::OnceLock};
use regex::{Regex};
use indexmap::IndexMap;

const NET_MERGES:u16=16126;
//NET_MERGES can never exceed 65,278 because u16 limits us to 65,536 total IDs.


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


fn read_files()->String{
    let folder_location="../dataset/arxiv_texts";
    let mut all_contents=String::new();

    let entries: fs::ReadDir=fs::read_dir(folder_location)
        .expect("ERROR IN READING DIRECTORY FILES.");

    for entry in entries.flatten(){
        let path=entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str())==Some("txt"){
            let contents=fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("FAILED TO READ FILES{:?}",path));
            all_contents.push_str(&contents);
            all_contents.push_str("<|endoftext|>\n\n\n");
        }
    }
    all_contents
}
//all_contents is a string with all the contents of the arxiv_texts(currently) to avoid them from being clumped up together
//i also push <|endoftext|> token and 3 new lines after each document.


fn count_pairs(tokenized_chunks:&[(Vec<u16>,usize)])->IndexMap<(u16,u16),usize>{

    let mut pair_counts=IndexMap::new();
    //This line creates a new indexmap being pair_counts. an indexmap is basically a hybrid data structure containing 2 things a standard HASHMAP and a Vector(in the form of array.)
    //A hashmap is basically a long dictionary which matches a key to value this is wht converts text to a numeric array index this is basically the adress for a computer
    //Reason for choosing IndexMap is logged in "decisions.md"
    for (chunk,freq) in tokenized_chunks{
        if chunk.len()<2{
            continue;
        }
    //This checks if the chunk even has 2 or more tokens to initite bpe or not.

        for pair in chunk.windows(2){
            let p=(pair[0],pair[1]);
            *pair_counts.entry(p).or_insert(0)+=*freq;
            //p is a tuple representing a specific pair of adjacent token IDs.
            //The other half of the expression acts as an automated tally sheet: it checks the map to see if pair p is already recorded, automatically writes it down with a starting score of 0 if it's brand new, and then immediately adds the chunk's frequency (*freq) to its running total.
        }
    }
    pair_counts
}

fn adjust_pairs(pair_count:&mut IndexMap<(u16,u16),usize>,pair:(u16,u16),delta:isize){
    //This part of the code safely updates pair counts.
    //When frequency of a token hits zero, completely removes the pair from the map to optimize memory and lookups.
    //This signifacantly fastens our bpe.
    if delta>0{
        *pair_count.entry(pair).or_insert(0)+=delta as usize;
    }else if delta<0 && let Some(c)=pair_count.get_mut(&pair){
            *c=c.saturating_sub((-delta) as usize);
            if *c==0{
                pair_count.swap_remove(&pair);
            }    
    }
}

fn split_text(text: &str)->Vec<&str>{
    // 'a is a rust lifetime label which gurantess that the string slices &str can never outlive orignal text.
    //This approach prevents rust from dangling pointers without sacrificing any runtime cost.

    static SPECIAL_REGEX:OnceLock<Regex>=OnceLock::new();
    static CLEAN_TEXT_REGEX:OnceLock<Regex>=OnceLock::new();

    let special_regex=SPECIAL_REGEX.get_or_init(|| 
        {Regex::new(r"<\|endoftext\|>|\[cite\]").unwrap()
    }); 
    let clean_text_way = CLEAN_TEXT_REGEX.get_or_init(|| 
        {Regex::new(r"<\|endoftext\|>|\[cite\]|\$| ?\p{L}+| ?\p{N}+|[^\s\p{L}\p{N}$]+|\s+").unwrap()
    });
    //These 2 specific line is the heart of the entire code. it tells the computer whr to chop the string into smaller units.
    //Because we needs to preserve $ and [cite] and hence they are given the priority and the code splits them first.
    //It is important to note that i did consider fancy-regex over normal regex as used in tiktoken to account for backtracking of tokens but considering the effeciency trade-off i went with standard regex.
    //As with standard regex we unlock Rust's Deterministic Finite Automaton (DFA) regex engine which processes text in strict, guaranteed O(n) linear time using SIMD making it significantly faster than fancy-regex. 

    let mut clean_text: Vec<&str>=Vec::new();
    let mut last_idx=0;

    for mat in special_regex.find_iter(text) {
        let text_segment=&text[last_idx..mat.start()];
        if !text_segment.is_empty(){
            for sub_mat in clean_text_way.find_iter(text_segment){
                clean_text.push(sub_mat.as_str());
            }
        }
        clean_text.push(mat.as_str());
        last_idx = mat.end();
    }

    let remaining_segment=&text[last_idx..];
    if !remaining_segment.is_empty(){
        for sub_mat in clean_text_way.find_iter(remaining_segment){
            clean_text.push(sub_mat.as_str());
        }
    }
    clean_text    
}

fn apply_merge(token:&[u16],target_pair:(u16, u16),new_id:u16)->Vec<u16>{
    let mut i =0;
    //An index pointer "i" is initialized at 0. It is responsible for tracking current token position as the code moves through the chunks.

    let mut merged_chunk=Vec::with_capacity(token.len());
    //This is the scratchpad vector for merging.
    //This is also making our bpe run significantly faster as normally rust has to frequently pause and reallocate a larger memory block.
    //But when we pre allocate the exact capacity required the vector writes its data instantly without waiting.            
    while i<token.len(){
            if i+1 < token.len() && token[i] == target_pair.0 && token[i+1]==target_pair.1{
                //This part acts as a conditional safety check checking if a neighbouring token even exists--i+1 < token.len()
                //And the second part checks if the current token and its neighbour perfectly match the winning pair--token[i] == best_pair.0 && token[i+1]==best_pair.1                        
                merged_chunk.push(new_id);
                i+=2;
                // i is incremented by 2 as 2 characters have been processed.
            }else {
                merged_chunk.push(token[i]);
                i+=1;
            }
        }   
    merged_chunk 
} 

fn bpe(text: &str){
    let clean_text=split_text(text);

    let mut chunk_counts:HashMap<Vec<u16>,usize>=HashMap::new();
    //This line creates tokenized_chunks which is basically a list inside another list, in simple terms a vector inside of vector.
    //It is used as bpe must strictly respect boundaries and it is forbidden from merging the end of one word with the beginning of new word.
    //This helps us sandbox everything in its private sandbox. 

    for chunk in &clean_text{
        let byte_tokens:Vec<u16>=if *chunk=="[cite]"{
            vec![256]
            //This line assigns the [cite] chunk the token number 256.
        }else if *chunk=="<|endoftext|>"{
            vec![257]
        }else{
            chunk.as_bytes().iter().map(|&b| b as u16).collect()
            //The function .as_bytes() performs an O(1) operation and treats string silence directly in u8 bytes
            //Because UTF-8 strings are already valid byte arrays in memory, this requires zero allocation or string copy.
            //Then we convert the u8 parts into u16 as u8 is very small ending at around 255.
            //The .collect() function is the most important part as it transforms an iterator---"chunk.as_bytes().iter().map(|&b| b as u16)",accomplished by .iter()
            //This iterator is then converted into a concrete collection.
            //It forces the iterator to execute,process the elements and gathers them in the desired data structure which is Vector in this context.
        };
        *chunk_counts.entry(byte_tokens).or_insert(0)+=1
    }

    let mut tokenized_chunks:Vec<(Vec<u16>,usize)>=chunk_counts.into_iter().collect();
    //To avoid processing duplicate text blocks repeatedly, 
    //I first groups identical token sequences into a HashMap alongside their occurrence frequencies.
    //I then converts that hash map into a vector of tuples (Vec<u16>, usize). 
    //This vector represents the entire text corpus broken down into unique, weighted token sequences.

    let mut next_token_id:u16=258;
    //We chose 258 for new merged tokens as avoids overlapping with any pre existing characters or special tokens.
    //ID(s)-
    //(0-255)->ASCII/UTF-8 characters
    //256,257->Assigned to [cite] and <|endoftext|>
    //258-desired number(limit is 65,535)->Assigned to merged tokens.

    let file=File::create("merges.bin")
        .expect("ERROR IN CREATING BINARY FILE.");

    let mut writer=BufWriter::new(file);
    //Instead of interacting directly with a hard drive or SSD for every single write operation BufWriter collects the bytes safely in a fast RAM memory buffer and only flushes them to the disk in large, optimized chunks at the very end.
    //This also makes our bpe incrementally faster.

    let mut pair_counts=count_pairs(&tokenized_chunks);
    for iteration in 0.. NET_MERGES{
        
        let best_option_pair: Option<(&(u16, u16), &usize)>=pair_counts.iter().max_by(|a,b|{
            // The .iter().max_by() part loops through the pair counts, comparing frequencies first, and uses secondary tie-breaking logic if frequencies match.
            //And finds the when in which the count is the highest

            let freq=a.1.cmp(b.1);
            freq.then_with(||b.0.cmp(a.0))
        });


        if let Some((best_pair,frequency))=best_option_pair{
            let best_pair=*best_pair;
            let frequency=*frequency;
            pair_counts.swap_remove(&best_pair);

            if iteration%500==0 || iteration==NET_MERGES-1{
                println!("ITERATION--{:2} MERGING PAIR--{:?} (FREQUENCY--{}) INTO TOKEN ID{}\n  |\n  |\n  |",
                iteration,best_pair,frequency,next_token_id);
            }
        writer.write_all(&best_pair.0.to_le_bytes())
        //Writes bytes 0-1.
            .expect("FAILED TO WRITE LEFT TOKEN BYTE.");
        writer.write_all(&best_pair.1.to_le_bytes())
        //Writes bytes 2-3.
            .expect("FAILED TO WRITE RIGHT TOKEN BYTE.");
        writer.write_all(&next_token_id.to_le_bytes())
        //Writes bytes 4-5.
            .expect("FAILED TO WRITE MERGED TOKEN BYTE.");                

        //ACTUAL IMPLEMENTATION OF THE BPE ALGORITHM (IMPLEMENTATION OF PAPER--"INCREMENTAL BPE TOKENIZATION")

            for (chunk, freq) in &mut tokenized_chunks{
                if !chunk.windows(2).any(|w| w[0]==best_pair.0 && w[1]==best_pair.1){
                    continue;
                }

                for pair in chunk.windows(2){
                    let p=(pair[0],pair[1]);
                    if p !=best_pair{
                        adjust_pairs(&mut pair_counts,p,-(*freq as isize));
                    }
                }

                let merged_chunk=apply_merge(chunk, best_pair,next_token_id);

                for pair in merged_chunk.windows(2){
                    adjust_pairs(&mut pair_counts,(pair[0],pair[1]),*freq as isize);
                }
                *chunk=merged_chunk;
                //This line derefrences the orignal chunk variable and completely replaces it with the newly compressed merged_chunk from scratchpad.
                //As it is impossible to modify an array safely(in this context--chunk) while reading it, we use a scratchpad(merged_chunk) which contains the new,updated sequence.
                //So after the bpe algorithm is finished we throw away *chunk and replace it with the new merged_chunk.

            }
            next_token_id+=1
        }else{
            println!("NO MORE PAIRS FOUND FOR MERGING. STOPPING EARLY");
            break;
        }
    }
    writer.flush()
        .expect("FAILED TO FLUSH DATA TO SSD.");
    println!("\nSUCCESS:6 BYTE PACKED RULES ARE WRITTEN IN THE BINARY FILE COMPLETELY.\n");
}


fn loader(file_path:&str)->Vec<((u16,u16),u16)> {
    let net_bytes=fs::read(file_path)
        .expect("FAILED TO READ BINARY FILE.");
    assert_eq!(net_bytes.len() % 6,
        0,
        "CORRUPTED BINARY FILE. FLIE SIZE({} BYTES) ISNT A MULTIPLE OF 6",
        net_bytes.len()
    );
    let mut merges=Vec::new();
    //This vector represents the entire history of all the iterations.

    for chunk in net_bytes.as_chunks::<6>().0{
        let left=u16::from_le_bytes(chunk[0..2].try_into().unwrap());
        //Grabs the first 2 chunks.

        let right=u16::from_le_bytes(chunk[2..4].try_into().unwrap());
        //Grabs the next 2 chunks.

        let merged_id=u16::from_le_bytes(chunk[4..6].try_into().unwrap());
        //Grabs the last 2 chunks.

        merges.push(((left,right),merged_id));
    }
    merges
}

fn build_vocab(merges:&[((u16,u16),u16)])->HashMap<u16,Vec<u8>>{
    let mut vocab:HashMap<u16,Vec<u8>>=HashMap::new();
    for i in 0..=255u16 {
        vocab.insert(i,vec![i as u8]);
        //To map token IDs back to human readable bytes, the vocab stores a Vec<u8> for every single token id.
    }
    vocab.insert(256,b"[cite]".to_vec());
    vocab.insert(257,b"<|endoftext|>".to_vec());
    
    for &((left,right),merged_id) in merges{
        let mut combined=vocab[&left].clone();
        combined.extend_from_slice(&vocab[&right]);
        // combined.extend_from_slice() is doing recursive reconstruction of text from merge history.
        //The loop is able to build vocab by gluing its left and right component byte vectors together.
        vocab.insert(merged_id,combined);
    }
    vocab
}


struct Tokenizer{
    vocab:HashMap<u16,Vec<u8>>,
    merges:HashMap<(u16,u16),u16>,
}

impl Tokenizer{
    fn new(merges_list: &[((u16,u16),u16)])->Tokenizer{
        Tokenizer{
            vocab:build_vocab(merges_list),
            merges:merges_list.iter().copied().collect(),
        }
    }
    
    fn encode(&self,text: &str)->Vec<u16>{
    let clean_text=split_text(text);

    let mut encoded_chunks:Vec<u16>=Vec::new();
    for chunk in &clean_text{
        if *chunk=="[cite]"{
            encoded_chunks.push(256);
            continue;
        }
        if *chunk=="<|endoftext|>"{
            encoded_chunks.push(257);
            continue;
        }

        let mut byte_tokens:Vec<u16>=chunk.as_bytes().iter().map(|&b| b as u16).collect();
        while byte_tokens.len()>=2{
            let mut best_merge:Option<((u16,u16),u16)>=None;
            for pair in byte_tokens.windows(2){
                let p=(pair[0],pair[1]);
                if let Some(&merged_id)=self.merges.get(&p)
                    && best_merge.is_none_or(|(_, best_id)|merged_id<best_id){
                        //best_merge is an option which originally starts as None and after the loop holds our best pair.
                        //This line handles the cases if best merge is none than it returns true.
                        //If best merge pair is something else it ignores the pair and runs merged_id<best_id and returns true only if the new pair has a smaller merged_id.
                        //Signifying it was learned during training.
                        best_merge=Some((p, merged_id));
                    }
            }
            let Some(((left_id,right_id),merged_id))=best_merge else{
                break;
            };
            byte_tokens=apply_merge(&byte_tokens, (left_id,right_id),merged_id);
        }
        encoded_chunks.extend(byte_tokens);
    }
    encoded_chunks
}

    fn decode(&self,ids:&[u16])->Result<String,String>{
        let mut bytes:Vec<u8>=Vec::with_capacity(ids.len()*4);
        //This line estimates that each BPE token expands to roughly 4 bytes (4 characters) of text on average.Even if off by little rust can expand the vector sometimes rather than everytime.

        for &id in ids{
            let token_bytes=self.vocab
                .get(&id)
                .ok_or_else(|| format!("UNKNOWN TOKEN ID IN DECODE {}",id))?;
            bytes.extend_from_slice(token_bytes);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
}

}
