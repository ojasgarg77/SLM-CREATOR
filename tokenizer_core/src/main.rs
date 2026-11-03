use std::fs;
use regex::Regex;

fn read_files()->String{
    let file_location="../dataset/arxiv_texts/2210.17432.txt";
    let contents=fs::read_to_string(file_location)
    .expect("ERROR IN READING FILES");
    contents
}


fn bpe(text: &str){
    let clean_text_way=Regex::new(r"\[cite\]|\$|\w+|\s+|[^\w\s]").unwrap();
    let mut clean_text:Vec<&str>=Vec::new();

    for mat in clean_text_way.find_iter(text){
        clean_text.push(mat.as_str());
    }

    let mut tokenized_chunks:Vec<Vec<u8>>=Vec::new();

    for chunk in &clean_text{
        let byte_tokens:Vec<u8>=chunk.as_bytes().to_vec();
        tokenized_chunks.push(byte_tokens);
    }

}

fn main(){
    let file_contents=read_files();
    bpe(&file_contents);
}