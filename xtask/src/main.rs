//! `cargo xtask <cmd>` — read-only dev loop for rusty-engine.
//!
//! Subcommands (B0): banner · fmt · clippy · test · check.
//! Band-gated verbs (`build-matrix`, `dist`, `loc`) are added in B6/B11.

use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("banner");
    match cmd {
        "banner" => println!("rusty-engine xtask: banner | fmt | clippy | test | check"),
        "fmt" => run("cargo", &["fmt", "--all"]),
        "clippy" => run("cargo", &["clippy", "--all-targets"]),
        "test" => run("cargo", &["test"]),
        "check" => {
            run("cargo", &["fmt", "--all"]);
            run("cargo", &["clippy", "--all-targets"]);
            run("cargo", &["test"]);
        }
        other => {
            eprintln!("unknown xtask command: {other}");
            std::process::exit(2);
        }
    }
}

fn run(prog: &str, args: &[&str]) {
    let status = Command::new(prog)
        .args(args)
        .status()
        .expect("failed to spawn cargo");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}
