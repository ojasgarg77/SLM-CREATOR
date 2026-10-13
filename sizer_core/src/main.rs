use std::io;

fn read_python_line()->String{
    let mut buf=String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("Failed to read line");
    buf.trim().to_string()
}

fn main() {
    let mut time_compute=read_python_line();
    let mut vram_compute=read_python_line();
    println!("{time_compute}/{vram_compute}");
}