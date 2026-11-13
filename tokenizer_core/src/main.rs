use std::{collections::HashMap, fs::{self, File}, io::{BufWriter,Write}};
use regex::Regex;
use indexmap::IndexMap;

const NET_MERGES:u16=5;
//NET_MERGES can never exceed 65,276 because u16 limits us to 65,535.

fn read_files()->String{
    let folder_location="../dataset/arxiv_texts";
    let mut all_contents=String::new();

    let entries: fs::ReadDir=fs::read_dir(folder_location)
    .expect("ERROR IN READING DIRECTORY FILES.");

    for entry in entries{
        if let Ok(entry) =entry{
            let path=entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str())==Some("txt"){
                let contents=fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("FAILED TO READ FILES{:?}",path));
                all_contents.push_str(&contents);
                all_contents.push_str("<|endoftext|>\n\n\n");
            }
        } 
    }
    all_contents
}
//all_contents is a string with all the contents of the arxiv_texts(currently) to avoid them from being clumped up together
//i also push 3 new lines after each line.


fn count_pairs(tokenized_chunks:&[Vec<u16>])->IndexMap<(u16,u16),usize>{

    let mut pair_counts=IndexMap::new();
    //This line creates a new indexmap being pair_counts. an indexmap is basically a hybrid data structure containing 2 things a standard HASHMAP and a Vector(in the form of array.)
    //A hashmap is basically a long dictionary which matches a key to value this is wht converts text to a numeric array index this is basically the adress for a computer
    //Reason for choosing IndexMap is logged in "decisions.md"
    for chunk in tokenized_chunks{
        if chunk.len()<2{
            continue;
        }
    //This checks if the chunk even has 2 or more tokens to initite bpe or not.

        for pair in chunk.windows(2){
            let p=(pair[0],pair[1]);
            *pair_counts.entry(p).or_insert(0)+=1;
    //This counts how many times every adjacent pair of tokens appear across text chunks.
    //here p is a tuple which stores the same. if it alrdy exists we dont count it. initializes to 0--if pair is new,and then increases count to 1.
        }
    }
    pair_counts
}


fn bpe(text: &str){
    let clean_text_way=Regex::new(r"<\|endoftext\|>|\[cite\]|\$|\w+|\s+|[^\w\s]").unwrap();
    //This specific line is the heart of the entire code. it tells the computer whr to chop the string into smaller units.
    //Because we needs to preserve $ and [cite] and hence they are given the priority and the code splits them first.
    let mut clean_text:Vec<&str>=Vec::new();

    for mat in clean_text_way.find_iter(text){
        //this purticular line applies the regex rule over the accumulated text.
        clean_text.push(mat.as_str());
        //the function .as_str() when applied to mat creates a fat pointer.
        //A FAT pointer is of 16 bytes it basically combines a standard memory address with additional metadata.
    }

    let mut tokenized_chunks:Vec<Vec<u16>>=Vec::new();
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
        tokenized_chunks.push(byte_tokens);
    }

    let mut next_token_id:u16=258;
    //We chose 258 for new merged tokens as avoids overlapping with any pre existing characters or special tokens.
    //ID(s)-
    //(0-255)->ASCII/UTF-8 characters
    //256,257->Assigned to [cite] and <|endoftext|>
    //258-desired number(limit is 65,535)->Assigned to merged tokens.

    let file=File::create("merges.bin")
        .expect("ERROR IN CREATING BINARY FILE.");

    let mut writer=BufWriter::new(file);

    for iteration in 0.. NET_MERGES{
        let pair_counts=count_pairs(&tokenized_chunks);
        
        let best_option_pair=pair_counts.iter().max_by(|a,b|{
            //The .iter.max_by_key() part loops through the entire database and ignores the token themselves and looks only at the count.
            //And finds the when in which the count is the highest

            let freq=a.1.cmp(b.1);
            freq.then_with(||b.0.cmp(a.0))
        });


        if let Some((best_pair,frequency))=best_option_pair{
            println!("ITERATION--{:2} MERGING PAIR--{:?} (FREQUENCY--{}) INTO TOKEN ID{}",
        iteration,best_pair,frequency,next_token_id);
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

            for chunk in &mut tokenized_chunks{
                let mut i =0;
                //An index pointer "i" is initialized at 0. It is responsible for tracking current token position as the code moves through the chunks.

                let mut merged_chunk=Vec::new();
                //This is the vector in which newly compressed version of the chunk is going to be built.

                while i<chunk.len(){
                    if i+1 < chunk.len() && chunk[i] == best_pair.0 && chunk[i+1]==best_pair.1{
                        //This part acts as a conditional safety check checking if a neighbouring token even exists--i+1 < chunk.len()
                        //And the second part checks if the current token and its neighbour perfectly match the winning pair--chunk[i] == best_pair.0 && chunk[i+1]==best_pair.1
                        merged_chunk.push(next_token_id);
                        i+=2;
                        // i is incremented by 2 as 2 characters have been processed.
                    }else {
                        merged_chunk.push(chunk[i]);
                        i+=1;
                    }
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
        writer.flush()
            .expect("FAILED TO FLUSH DATA TO SSD.");
    }
    println!("\nSUCCESS:6 BYTE PACKED RULES ARE WRITTEN IN THE BINARY FILE COMPLETELY.\n");
}

fn loader(file_path:&str)->Vec<((u16,u16),u16)> {
    let net_bytes=fs::read(file_path)
        .expect("FAILED TO READ BINARY FILE.");
    let mut merges=Vec::new();

    for chunk in net_bytes.chunks_exact(6){
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
    }
    vocab.insert(256,b"[cite]".to_vec());
    vocab.insert(257,b"<|endoftext|>".to_vec());
    
    for &((left,right),merged_id) in merges{
        let mut combined=vocab[&left].clone();
        combined.extend_from_slice(&vocab[&right]);
        vocab.insert(merged_id,combined);
    }
    vocab
}

fn main(){
    let file_contents=read_files();
    bpe(&file_contents);

    let merges=loader("merges.bin");
    println!("LOADED {} MERGE RULES.",merges.len());

    let vocab=build_vocab(&merges);
}