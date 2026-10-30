use std::{io};
use std::process;

const MIB_TO_BYTES:u64=1_048_576;
const BYTES_PER_PARAM:u64=20;
const TFLOPS_EFFICIENCY_FACTOR:f64=0.3;
const MAX_PARAMS:f64=100_000_000.0;
const EPOCHS:u64=4;
const TOKEN_PER_PARAM:u64=20;

//EPOCHS are hardcoded because study shows models train best at 4 EPOCHS only.On adding more models tend to memorize the text instead of learning.
// and if not given enough the model tends to outputs glibberish.

fn read_python_line()->String{
    let mut buf=String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("Failed to read line");
    buf.trim().to_string()
}

fn main() {
    let mut limiting_factors=vec![];
    limiting_factors.push(MAX_PARAMS);

    let vram_computed_params=params_from_vram();
    limiting_factors.push(vram_computed_params as f64);

    let time_computed_params=params_from_time();
    limiting_factors.push(time_computed_params);

    let token_computed_params:u64=params_from_tokens();
    limiting_factors.push(token_computed_params as f64);

    let min_limiting_factor=limiting_factors.iter()
    .min_by(|a,b| a.partial_cmp(b).unwrap())
    .copied()
    .unwrap_or(0.0);
    println!("{}",min_limiting_factor as u64)

}

fn params_from_vram()->u64{
    let vram_compute=read_python_line();
    let vram_compute:u64=match vram_compute.parse(){
        Ok(num)=>num,
        Err(_)=>{
            eprintln!("Failed to parse vram string.");
            process::exit(1); 
        }
    };
    let mut vram_bytes=vram_compute*MIB_TO_BYTES;
    vram_bytes=(90u64*vram_bytes)/100u64;
    let vram_params=vram_bytes/BYTES_PER_PARAM;
    vram_params
}

fn params_from_time()->f64{
    let time_compute=read_python_line();
    let tflops_compute=read_python_line();
    let tflops_compute:f64=match tflops_compute.parse(){
        Ok(num)=>num,
        Err(_)=>{
            eprintln!("Failed to parse TFLOPS string.");
            process::exit(1); 
        }
    };
    let time_compute:f64=match time_compute.parse(){
        Ok(num)=>num,
        Err(_)=>{
            eprintln!("Failed to parse time string.");
            process::exit(1); 
        }
    }; 
    let total_seconds=time_compute*3600.0;
    let total_flops=tflops_compute*1_000_000_000_000.0*total_seconds*TFLOPS_EFFICIENCY_FACTOR;
    let time_params=(total_flops/120.0).sqrt();
    time_params
}
// In deep learning a transformer model requires 6 flops per param for every token.
// The formula for training=> total flops=6*model params*total no. of training tokens.
// According to the chinchilla paper a model works best when=> total no. of training tokens=20*model_params.
// Hence the used formula in my code is going to be n(total no. of params)= (flops/120) to the power of half.

fn params_from_tokens()->u64{
    let tokens_compute=read_python_line();
    let tokens_compute:u64=match tokens_compute.parse(){
        Ok(num)=>num,
        Err(_)=>{
            eprintln!("Failed to parse token string.");
            process::exit(1); 
        }
    };

    if tokens_compute==0{
            eprintln!("No training tokens. PLZ check the dataset folder once.");
            process::exit(1);         
    }

    let token_params=EPOCHS*tokens_compute/TOKEN_PER_PARAM;
    token_params
}