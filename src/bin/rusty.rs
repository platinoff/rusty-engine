//! `rusty` — single-binary entry point (produces rusty.exe / rusty / rusty.apk
//! per band B6). B0 dispatches identity commands; `serve`/`node`/`hub` land in B5/B8.

use std::env;

fn main() {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("--version") | Some("-V") => println!("{}", rusty_engine::VERSION),
        Some("--help") | Some("-h") | None => {
            println!("{}", rusty_engine::banner());
            println!("usage: rusty [subcommand]");
            println!();
            println!("  --version, -V   print crate version");
            println!("  --help, -h      print this banner");
            println!("  inspect <file>  GGUF metadata summary     (band B1)");
            println!("  run -m <model>  generate text            (band B3)");
            println!("  serve           OpenAI-compat server     (band B5)");
            println!("  node            Wi-Fi slave runtime      (band B7)");
            println!("  hub             GSV master supervisor    (band B8)");
        }
        Some(cmd) => {
            println!(
                "{}: subcommand '{cmd}' is scaffolded (bands B1+)",
                rusty_engine::banner()
            );
        }
    }
}
