// src/main.rs
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Dummy {
    id: u32,
}

fn main() {
    let dummy = Dummy { id: 1 };
    let json = serde_json::to_string(&dummy).unwrap();
    println!("{}", json);
}
