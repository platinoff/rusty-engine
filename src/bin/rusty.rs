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
            println!("  inspect <file> [N]        GGUF metadata summary      (PH-107)");
            println!("  tensor <file> <name>      mmap + dequant one tensor  (PH-305a)");
            println!("  config <file>             read model hyperparameters (PH-305c)");
            println!("  validate <file>           check tensor names/dims vs config (PH-305e)");
            println!("  run -m <model> -p <text>  BPE text generation        (PH-306b)");
            println!("  run -m <model> -t <ids>   numeric-token greedy       (PH-306)");
            println!(
                "  serve [--port N] [--host H]  OpenAI-compat HTTP, LAN-first :8090 (PH-501a/507)"
            );
            println!("  heartbeat                 stamp GSV keep-live file   (PH-504)");
            println!("  node                      Wi-Fi slave runtime        (band B7)");
            println!("  hub                       GSV master supervisor      (band B8)");
            println!(
                "  worker [--live]           hub-edge worker/APK role (dry-run default, 2026-10-16)"
            );
        }
        Some("inspect") => match args.next() {
            Some(path) => inspect(&path, args.next().as_deref()),
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
        Some("run") => run(args),
        Some("serve") => serve_cmd(args),
        Some("heartbeat") => heartbeat_cmd(),
        Some("worker") => worker_cmd(args),
        Some(other) => {
            println!(
                "{}: subcommand '{other}' is scaffolded (bands B1+)",
                rusty_engine::banner()
            );
        }
    }
}

/// Compact one-line rendering of a GGUF KV value: scalars as-is, strings truncated
/// (chat templates run to kilobytes), arrays as `kind[len]` with inline contents only
/// for short numeric arrays (`PH-1201` vision-layout spike).
fn kv_summary(v: &rusty_engine::gguf::Value) -> String {
    use rusty_engine::gguf::Value;
    match v {
        Value::Str(s) => {
            if s.chars().count() > 88 {
                let head: String = s.chars().take(88).collect();
                format!("\"{head}…\" ({} chars)", s.chars().count())
            } else {
                format!(
                    "\"{}\"",
                    s.replace('\\', "\\\\")
                        .replace('\n', "\\n")
                        .replace('\t', "\\t")
                )
            }
        }
        Value::Array(kind, vals) => {
            let short = vals.len() <= 8
                && !vals.is_empty()
                && vals
                    .iter()
                    .all(|x| !matches!(x, Value::Str(_) | Value::Array(_, _)));
            if short {
                let items: Vec<String> = vals.iter().map(kv_summary).collect();
                format!("{kind:?}[{}] = [{}]", vals.len(), items.join(", "))
            } else {
                format!("{kind:?}[{}]", vals.len())
            }
        }
        other => format!("{other:?}"),
    }
}

/// Print the GGUF header: counts, alignment, architecture, every KV key (abbreviated)
/// with a tensor-name prefix histogram, and the first tensor rows. The KV dump + histogram
/// are the `PH-1201` vision-layout spike: what a container carries hangs on the non-qwen2
/// keys and the name prefixes, not on the first ten tensor lines.
fn inspect(path: &str, show_n: Option<&str>) {
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
            println!("  metadata:");
            for (k, v) in &f.metadata {
                println!("    {k} = {}", kv_summary(v));
            }
            let mut hist: std::collections::BTreeMap<String, (usize, u64)> =
                std::collections::BTreeMap::new();
            for t in &f.tensors {
                let root = t.name.split('.').next().unwrap_or("").to_string();
                let e = hist.entry(root).or_default();
                e.0 += 1;
                e.1 += t.nbytes().unwrap_or(0) as u64;
            }
            println!("  tensor-name prefixes:");
            for (p, (n, bytes)) in &hist {
                println!(
                    "    {p:<30} {n:>5} tensors  {:>9.1} MiB",
                    *bytes as f64 / (1 << 20) as f64
                );
            }
            let show = match show_n {
                Some(n) => n.parse::<usize>().unwrap_or(10).min(f.tensors.len()),
                None => f.tensors.len().min(10),
            };
            println!("  tensors (showing {show} of {}):", f.tensors.len());
            for t in f.tensors.iter().take(show) {
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

/// `rusty run -m <model.gguf> (-p "text" | -t "0,1,2") [--max-tokens N]` — dense greedy
/// generation. `-p` runs the BPE text pipeline (PH-306b); `-t` uses numeric token ids (PH-306).
fn run<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut model_path: Option<String> = None;
    let mut tokens: Vec<u32> = Vec::new();
    let mut prompt: Option<String> = None;
    let mut max_new = 8usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next(),
            "-p" | "--prompt" => prompt = args.next(),
            "-t" | "--tokens" => {
                if let Some(spec) = args.next() {
                    tokens = spec
                        .split(',')
                        .map(|x| x.trim())
                        .filter(|x| !x.is_empty())
                        .map(|x| {
                            x.parse::<u32>().unwrap_or_else(|_| {
                                eprintln!("rusty run: bad token id '{x}'");
                                std::process::exit(2)
                            })
                        })
                        .collect();
                }
            }
            "--max-tokens" => {
                max_new = args.next().map_or(8, |v| {
                    v.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("rusty run: bad --max-tokens");
                        std::process::exit(2)
                    })
                })
            }
            other => {
                eprintln!("rusty run: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let path = match model_path {
        Some(p) if !p.is_empty() => p,
        _ => {
            eprintln!(
                "usage: rusty run -m <model.gguf> (-p \"text\" | -t \"0,1,2\") [--max-tokens N]"
            );
            std::process::exit(2);
        }
    };
    let model = match unsafe { rusty_engine::loader::MmapModel::open(Path::new(&path)) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rusty run {path}: {e}");
            std::process::exit(1);
        }
    };
    // Text pipeline (PH-306b): needs tokenizer.ggml.* meta.
    if let Some(text) = prompt {
        let tr = match rusty_engine::textgen::TextRunner::from_model(&model) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("rusty run {path}: {e}");
                std::process::exit(1);
            }
        };
        match tr.run_text(&text, max_new) {
            Ok(got) => println!("[{}] {}", tr.runner.n_layers, got),
            Err(e) => {
                eprintln!("rusty run {path}: {e}");
                std::process::exit(1);
            }
        }
        std::process::exit(0);
    }
    if tokens.is_empty() {
        eprintln!("rusty run: no prompt (use -p \"text\" or -t \"id,id,...\")");
        std::process::exit(2);
    }
    let cfg = match rusty_engine::model::ModelConfig::from_file(model.header()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("rusty run {path}: config: {e}");
            std::process::exit(1);
        }
    };
    if let Some(prob) = cfg.validate(model.header()).first() {
        eprintln!("rusty run {path}: geometry invalid: {prob}");
        std::process::exit(1);
    }
    let runner = match rusty_engine::runner::DenseRunner::from_model(&model, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rusty run {path}: runner: {e}");
            std::process::exit(1);
        }
    };
    let out = runner.generate(&tokens, max_new);
    let ids: Vec<String> = out.iter().map(|t| t.to_string()).collect();
    println!(
        "[{}] {} layers — prompt {:?} -> {}",
        cfg.arch,
        cfg.n_layers,
        tokens,
        ids.join(",")
    );
    std::process::exit(0);
}

/// `rusty serve [--port N] [--host H]` — pump `crate::http::route` (PH-501a).
/// Band 233 (PH-507): LAN-first — binds `0.0.0.0` by default like `llama_serve`,
/// so the GSV hub and Wi-Fi/edge peers address it by the machine's local address;
/// `--host 127.0.0.1` keeps the old loopback-only posture. Serves one connection
/// per request.
fn serve_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut port = 8090u16;
    let mut host = "0.0.0.0".to_string();
    let mut model_path: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" | "-p" => {
                port = args.next().map_or(8090, |v| {
                    v.parse::<u16>().unwrap_or_else(|_| {
                        eprintln!("rusty serve: bad --port");
                        std::process::exit(2)
                    })
                })
            }
            "--host" => host = args.next().unwrap_or_else(|| "0.0.0.0".to_string()),
            "--model" | "-m" => model_path = args.next(),
            other => {
                eprintln!("rusty serve: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let addr = format!("{host}:{port}");
    println!(
        "{} — serving on http://{addr} (Ctrl-C to stop)",
        rusty_engine::banner()
    );
    // With a model, wire real BPE→generate→decode (PH-501b); else the stub route.
    let result = match &model_path {
        Some(p) => match unsafe { rusty_engine::loader::MmapModel::open(Path::new(p)) } {
            Ok(m) => match rusty_engine::textgen::TextRunner::from_model(&m) {
                Ok(tr) => {
                    println!("  model: {p} ({} layers)", tr.runner.n_layers);
                    rusty_engine::serve::serve_with_model(&addr, &tr)
                }
                Err(e) => {
                    eprintln!("rusty serve: model load: {e}");
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("rusty serve: open {p}: {e}");
                std::process::exit(1);
            }
        },
        None => rusty_engine::serve::serve(&addr),
    };
    if let Err(e) = result {
        eprintln!("rusty serve {addr}: {e}");
        std::process::exit(1);
    }
    std::process::exit(0);
}

/// `rusty worker [--live] [--peer-id ID] [--hub URL] [--serve URL] [--model ID]
/// [--token-env VAR] [--memory-mb N]` — the hub-edge worker/APK role (2026-10-16).
/// Dry-run by default: prints the register/join/heartbeat/poll plan (token masked) with
/// no socket. `--live` runs the blocking register → join → heartbeat/poll loop against the
/// GSV hub `/api/edge` (which fronts poolAI `/api/v1`; never `:8091` off-box).
fn worker_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut live = false;
    let mut peer_id = "rusty-01".to_string();
    let mut hub: Option<String> = None;
    let mut serve_url: Option<String> = None;
    let mut model = "rusty-05".to_string();
    let mut token_env = "GSV_EDGE_TOKEN".to_string();
    let mut mem_mb = 0usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--live" => live = true,
            "--peer-id" => peer_id = args.next().unwrap_or(peer_id),
            "--hub" => hub = args.next(),
            "--serve" => serve_url = args.next(),
            "--model" | "-m" => model = args.next().unwrap_or(model),
            "--token-env" => token_env = args.next().unwrap_or(token_env),
            "--memory-mb" => mem_mb = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            other => {
                eprintln!("rusty worker: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let token = env::var(&token_env).unwrap_or_default();
    let mut cfg = rusty_engine::worker::WorkerConfig::local(peer_id, token);
    if let Some(h) = hub {
        cfg.hub = h;
    }
    if let Some(s) = serve_url {
        cfg.serve_url = s;
    }
    cfg.model = model;
    if mem_mb > 0 {
        cfg.max_memory_mb = mem_mb;
    }
    println!(
        "{} — worker ({})",
        rusty_engine::banner(),
        if live { "LIVE" } else { "dry-run" }
    );
    if !live {
        for line in rusty_engine::worker::plan_lines(&cfg, cfg.max_memory_mb) {
            println!("  {line}");
        }
        println!("  (dry-run: no sockets; add --live to register + poll)");
        std::process::exit(0);
    }
    match rusty_engine::worker::join_once(&cfg, cfg.max_memory_mb) {
        Ok(_) => println!(
            "  registered as {} ({})",
            cfg.peer_id,
            rusty_engine::worker::ORIGIN
        ),
        Err(e) => {
            eprintln!("rusty worker: {e}");
            std::process::exit(1);
        }
    }
    let hb_every = (rusty_engine::worker::HEARTBEAT_SECS / rusty_engine::worker::POLL_SECS).max(1);
    let mut tick = 0u64;
    loop {
        if tick % hb_every == 0 {
            match rusty_engine::worker::heartbeat_once(&cfg, cfg.max_memory_mb) {
                Ok(st) if (200..300).contains(&st) => {}
                Ok(st) => eprintln!("rusty worker: heartbeat HTTP {st}"),
                Err(e) => eprintln!("rusty worker: heartbeat {e}"),
            }
        }
        match rusty_engine::worker::poll_once(&cfg, true) {
            Ok(line) if !line.is_empty() => println!("  {line}"),
            Ok(_) => {}
            Err(e) => eprintln!("rusty worker: poll {e}"),
        }
        tick += 1;
        std::thread::sleep(std::time::Duration::from_secs(
            rusty_engine::worker::POLL_SECS,
        ));
    }
}

/// `rusty heartbeat` — write `target/live/rusty_heartbeat.json` once for the GSV keep-live
/// hub mirror (PH-504). Uptime is 0 from a one-shot CLI; `rusty serve` keeps it current.
fn heartbeat_cmd() -> ! {
    match rusty_engine::hub::write_heartbeat(
        Path::new("target/live"),
        &rusty_engine::hub::Heartbeat::new(0),
    ) {
        Ok(path) => println!(
            "{} — heartbeat -> {}",
            rusty_engine::banner(),
            path.display()
        ),
        Err(e) => {
            eprintln!("rusty heartbeat: {e}");
            std::process::exit(1);
        }
    }
    std::process::exit(0);
}
