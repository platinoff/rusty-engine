//! `rusty` — single-binary entry point (produces rusty.exe / rusty / rusty.apk
//! per band B6). B0 identity + `inspect` (PH-107) + `tensor` (PH-305a real-tensor
//! load/dequant); `run`/`serve`/`node`/`hub` land in B3/B5/B7/B8.

use std::env;
use std::path::Path;

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next();
    match cmd.as_deref() {
        Some("--version") | Some("-V") => println!("{}", rusty_engine::VERSION),
        Some("--help") | Some("-h") | None => {
            println!("{}", rusty_engine::banner());
            println!("usage: rusty [subcommand]");
            println!();
            println!("  --version, -V   print crate version");
            println!("  --help, -h      print this banner");
            println!("  inspect <file>            GGUF metadata summary      (PH-107)");
            println!("  tensor <file> <name>      mmap + dequant one tensor  (PH-305a)");
            println!("  config <file>             read model hyperparameters (PH-305c)");
            println!("  validate <file>           check tensor names/dims vs config (PH-305e)");
            println!("  run -m <model>            generate text              (band B3)");
            println!("  serve                     OpenAI-compat server       (band B5)");
            println!("  node                      Wi-Fi slave runtime        (band B7)");
            println!("  hub                       GSV master supervisor      (band B8)");
        }
        Some("inspect") => match args.next() {
            Some(path) => inspect(&path),
            None => {
                eprintln!("usage: rusty inspect <model.gguf>");
                std::process::exit(2);
            }
        },
        Some("tensor") => match (args.next(), args.next()) {
            (Some(path), Some(name)) => tensor(&path, &name),
            _ => {
                eprintln!("usage: rusty tensor <model.gguf> <tensor.name>");
                std::process::exit(2);
            }
        },
        Some("config") => match args.next() {
            Some(path) => config(&path),
            None => {
                eprintln!("usage: rusty config <model.gguf>");
                std::process::exit(2);
            }
        },
        Some("validate") => match args.next() {
            Some(path) => validate(&path),
            None => {
                eprintln!("usage: rusty validate <model.gguf>");
                std::process::exit(2);
            }
        },
        Some(other) => {
            println!(
                "{}: subcommand '{other}' is scaffolded (bands B1+)",
                rusty_engine::banner()
            );
        }
    }
}

/// Print the GGUF header: counts, alignment, architecture and the first tensor rows.
fn inspect(path: &str) {
    match rusty_engine::gguf::parse_file(Path::new(path), rusty_engine::gguf::INSPECT_WINDOW) {
        Ok(f) => {
            println!(
                "{path}  [GGUF v{}]  {} tensors  {} kv  align {}",
                f.version,
                f.n_tensors,
                f.n_kv(),
                f.alignment()
            );
            if let Some(rusty_engine::gguf::Value::Str(arch)) = f.get("general.architecture") {
                println!("  architecture: {arch}");
            }
            let show = f.tensors.len().min(10);
            println!("  tensors (showing {show} of {}):", f.tensors.len());
            for t in f.tensors.iter().take(10) {
                println!(
                    "    {:<40} {:?} ggml={} dtype={:?} offset={}",
                    t.name, t.dims, t.ggml_type, t.dtype, t.offset
                );
            }
        }
        Err(e) => {
            eprintln!("rusty inspect {path}: {e}");
            std::process::exit(1);
        }
    }
}

/// mmap a real GGUF and materialize one tensor: byte-range bounds check + scalar dequant.
/// The first step of the forward path (`PH-305`) — reads real quantized weights, not synth.
fn tensor(path: &str, name: &str) {
    let model = match unsafe { rusty_engine::loader::MmapModel::open(Path::new(path)) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rusty tensor {path}: {e}");
            std::process::exit(1);
        }
    };
    let entry = match model.header().tensors.iter().find(|t| t.name == name) {
        Some(t) => t,
        None => {
            eprintln!(
                "rusty tensor: tensor '{name}' not found in {} tensors",
                model.header().tensors.len()
            );
            std::process::exit(1);
        }
    };
    println!(
        "'{name}'  dims={:?}  ggml={}  dtype={:?}  numel={}  nbytes={:?}  offset={}",
        entry.dims,
        entry.ggml_type,
        entry.dtype,
        entry.numel(),
        entry.nbytes(),
        entry.offset
    );
    match model.tensor_bytes(name) {
        Some(b) => println!(
            "  bounds-ok: {} bytes within file (data_start {})",
            b.len(),
            model.data_start()
        ),
        None => {
            eprintln!("  out-of-bounds or unsized type");
            std::process::exit(1);
        }
    }
    // Materialize floats: quantized via the oracle, F32 read directly.
    if entry.ggml_type == 0 {
        if let Some(b) = model.tensor_bytes(name) {
            let floats: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let head = &floats[..floats.len().min(6)];
            println!("  F32 {} values, first {head:?}", floats.len());
        }
    } else {
        match model.dequant(name) {
            Ok(v) => {
                let finite = v.iter().filter(|x| x.is_finite()).count();
                let head = &v[..v.len().min(6)];
                println!(
                    "  dequant {} values ({finite} finite), first {head:?}",
                    v.len()
                );
            }
            Err(e) => eprintln!("  dequant: {e}"),
        }
    }
}

/// Read model hyperparameters (PH-305c) from a GGUF header.
fn config(path: &str) {
    match rusty_engine::gguf::parse_file(Path::new(path), rusty_engine::gguf::INSPECT_WINDOW) {
        Ok(f) => match rusty_engine::model::ModelConfig::from_file(&f) {
            Ok(c) => {
                println!(
                    "{path}  [{}]  {} layers  d_model={}  head_dim={}",
                    c.arch,
                    c.n_layers,
                    c.d_model,
                    c.head_dim()
                );
                println!(
                    "  heads={} kv_heads={} d_ff={} vocab={} moe={}",
                    c.n_heads,
                    c.n_kv_heads,
                    c.d_ff,
                    c.n_vocab,
                    c.is_moe()
                );
                println!(
                    "  rope_base={} rms_eps={:?} experts={}(used {})",
                    c.rope_freq_base, c.rms_eps, c.n_experts, c.n_experts_used
                );
            }
            Err(e) => {
                eprintln!("rusty config {path}: {e}");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("rusty config {path}: {e}");
            std::process::exit(1);
        }
    }
}

/// Check that every tensor the forward loop needs is present with the right dims (PH-305e).
fn validate(path: &str) {
    let f =
        match rusty_engine::gguf::parse_file(Path::new(path), rusty_engine::gguf::INSPECT_WINDOW) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("rusty validate {path}: {e}");
                std::process::exit(1);
            }
        };
    let cfg = match rusty_engine::model::ModelConfig::from_file(&f) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rusty validate {path}: {e}");
            std::process::exit(1);
        }
    };
    match cfg.validate(&f) {
        problems if problems.is_empty() => {
            println!(
                "{path}: OK — [{}] {} layers, all expected tensors present with matching dims",
                cfg.arch, cfg.n_layers
            );
        }
        problems => {
            println!("{path}: {} problem(s):", problems.len());
            for p in problems.iter().take(30) {
                println!("  - {p}");
            }
            std::process::exit(1);
        }
    }
}
