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
            println!("  manifest <file> [--sha256] GGUF JSON manifest (read-only, GASV)");
            println!("  skeleton <spec.json> <out.gguf>  deterministic GGUF skeleton (GASV-C2)");
            println!("  transplant-mirror <skeleton.gguf> <tensors.json> <chunks_dir> <out.gguf>  direct-copy mirror tensors into skeleton (GASV-D1)");
            println!("  quantize-export <in.gguf> <out.gguf> <q4_0|q8_0|q4_k|q4_k_m> [--report <path>]  quantize F32 GGUF (GASV-D3/F2/G6; q4_k_m=no-imatrix fallback)");
            println!("  assemble-copy <src.gguf> <out.gguf> [--report <path>] [--kv k=v]...  stream-copy all tensors + metadata (GASV-G9 Strategy C)");
            println!("  artifact-chunks <src.gguf> <tensor> [--max N]  emit artifact-source chunk manifest (key/offset/len, GASV-G20)");
            println!("  artifact-report <file.gguf> [--kind K] [--runnable true|false] [--source S] [--coverage C] [--transplant-report-sha256 H]  GASV artifact contract (F1)");
            println!("  tensor <file> <name>      mmap + dequant one tensor  (PH-305a)");
            println!("  dump-f32 <file> <name> <out.bin>  raw LE f32 bytes of one tensor   (MC-2)");
            println!("  config <file>             read model hyperparameters (PH-305c)");
            println!("  validate <file>           check tensor names/dims vs config (PH-305e)");
            println!("  tokenize -m <f> -p <txt>  BPE token ids, no weights      (PH-1305)");
            println!("  run -m <model> -p <text>  BPE text generation        (PH-306b)");
            println!("  run|serve --vision-prefix <f32> [--vision-at N]  inject GSV Q4 vision vectors after N prompt tokens (PH-1204)");
            println!("  run -m <model> -t <ids>   numeric-token greedy       (PH-306)");
            println!("  audit -m <model> -p <txt> ActivationAudit stats (json) (ECO-9.2 9.3a)");
            println!("  pack-mint -m <model> -t <tensor> -b <idx>|--all [--zero]  WSP pack lines (stdout) (ECO-9.2 9.3b)");
            println!(
                "  serve [--port N] [--host H] [--patch <wsp>]  OpenAI-compat HTTP, LAN-first :8092 (PH-501a/507, EN-8; :8090 is rebook view)"
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
        Some("manifest") => {
            let mut path: Option<String> = None;
            let mut sha256 = false;
            for a in args.by_ref() {
                match a.as_str() {
                    "--sha256" => sha256 = true,
                    p if path.is_none() => path = Some(p.to_string()),
                    _ => {
                        eprintln!("usage: rusty manifest <model.gguf> [--sha256]");
                        std::process::exit(2);
                    }
                }
            }
            match path {
                Some(p) => manifest(&p, sha256),
                None => {
                    eprintln!("usage: rusty manifest <model.gguf> [--sha256]");
                    std::process::exit(2);
                }
            }
        }
        Some("skeleton") => match (args.next(), args.next()) {
            (Some(spec), Some(out)) => skeleton(&spec, &out),
            _ => {
                eprintln!("usage: rusty skeleton <spec.json> <out.gguf>");
                std::process::exit(2);
            }
        },
        Some("transplant-mirror") => match (args.next(), args.next(), args.next(), args.next()) {
            (Some(skel), Some(tj), Some(chunks), Some(out)) => {
                transplant_mirror(&skel, &tj, &chunks, &out)
            }
            _ => {
                eprintln!("usage: rusty transplant-mirror <skeleton.gguf> <tensors.json> <chunks_dir> <out.gguf>");
                std::process::exit(2);
            }
        },
        Some("artifact-chunks") => {
            let inp = args.next();
            let tensor = args.next();
            let mut max = 4usize * 1024 * 1024;
            while let Some(a) = args.next() {
                if a == "--max" {
                    max = args
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(4 * 1024 * 1024);
                }
            }
            match (inp, tensor) {
                (Some(i), Some(t)) => artifact_chunks(&i, &t, max),
                _ => {
                    eprintln!("usage: rusty artifact-chunks <src.gguf> <tensor> [--max N]");
                    std::process::exit(2);
                }
            }
        }
        Some("quantize-export") => {
            let inp = args.next();
            let out = args.next();
            let q = args.next();
            let mut report: Option<String> = None;
            while let Some(a) = args.next() {
                if a == "--report" {
                    report = args.next();
                }
            }
            match (inp, out, q) {
                (Some(i), Some(o), Some(qt)) => quantize_export(&i, &o, &qt, report.as_deref()),
                _ => {
                    eprintln!("usage: rusty quantize-export <in.gguf> <out.gguf> <q4_0|q8_0|q4_k|q4_k_m> [--report <path>]");
                    std::process::exit(2);
                }
            }
        }
        Some("assemble-copy") => {
            let inp = args.next();
            let out = args.next();
            let mut report: Option<String> = None;
            let mut add_kv: Vec<(String, String)> = Vec::new();
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--report" => report = args.next(),
                    "--kv" => {
                        if let Some(kv) = args.next() {
                            if let Some((k, v)) = kv.split_once('=') {
                                add_kv.push((k.to_string(), v.to_string()));
                            }
                        }
                    }
                    _ => {}
                }
            }
            match (inp, out) {
                (Some(i), Some(o)) => assemble_copy(&i, &o, report.as_deref(), &add_kv),
                _ => {
                    eprintln!("usage: rusty assemble-copy <src.gguf> <out.gguf> [--report <path>] [--kv key=value]...");
                    std::process::exit(2);
                }
            }
        }
        Some("artifact-report") => {
            let path = match args.next() {
                Some(p) => p,
                None => {
                    eprintln!("usage: rusty artifact-report <file.gguf> [--kind K] [--runnable true|false] [--source S] [--coverage C] [--transplant-report-sha256 H]");
                    std::process::exit(2);
                }
            };
            let mut kind = "partial-shexp-pack".to_string();
            let mut runnable = false;
            let mut source = String::new();
            let mut coverage = String::new();
            let mut transplant_sha = String::new();
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--kind" => {
                        kind = args.next().unwrap_or(kind);
                    }
                    "--runnable" => {
                        runnable = args.next().as_deref() == Some("true");
                    }
                    "--source" => {
                        source = args.next().unwrap_or_default();
                    }
                    "--coverage" => {
                        coverage = args.next().unwrap_or_default();
                    }
                    "--transplant-report-sha256" => {
                        transplant_sha = args.next().unwrap_or_default();
                    }
                    _ => {}
                }
            }
            artifact_report(&path, &kind, runnable, &source, &coverage, &transplant_sha);
        }
        Some("dump-f32") => match (args.next(), args.next(), args.next()) {
            (Some(path), Some(name), Some(out)) => dump_f32(&path, &name, &out),
            _ => {
                eprintln!("usage: rusty dump-f32 <model.gguf> <tensor.name> <out.bin>");
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
        Some("tokenize") => tokenize_cmd(args),
        Some("run") => run(args),
        Some("audit") => audit_cmd(args),
        Some("pack-mint") => pack_mint_cmd(args),
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

fn manifest(path: &str, with_sha256: bool) {
    let p = Path::new(path);
    let file_size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let sha256 = if with_sha256 {
        match rusty_engine::patch::sha256_hex_file(p) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("rusty manifest: sha256 failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    match rusty_engine::gguf::parse_file(p, rusty_engine::gguf::INSPECT_WINDOW) {
        Ok(f) => {
            let tensors: Vec<serde_json::Value> = f
                .tensors
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "dims": t.dims,
                        "ggml_type": t.ggml_type,
                        "dtype": t.dtype.map(|d| format!("{d:?}")),
                        "offset": t.offset,
                        "nbytes": t.nbytes(),
                        "numel": t.numel(),
                    })
                })
                .collect();
            let metadata: Vec<serde_json::Value> = f
                .metadata
                .iter()
                .map(|(k, v)| serde_json::json!({"key": k, "value": kv_summary(v)}))
                .collect();
            let out = serde_json::json!({
                "schema": "gasv.gguf-manifest.v1",
                "file": path,
                "file_size_bytes": file_size,
                "sha256": sha256,
                "gguf_version": f.version,
                "n_tensors": f.n_tensors,
                "n_kv": f.n_kv(),
                "alignment": f.alignment(),
                "data_offset": f.data_offset,
                "architecture": f.get("general.architecture").and_then(|v| match v { rusty_engine::gguf::Value::Str(s) => Some(s.as_str()), _ => None }),
                "metadata": metadata,
                "tensors": tensors,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".to_string())
            );
        }
        Err(e) => {
            eprintln!("rusty manifest {path}: {e}");
            std::process::exit(1);
        }
    }
}

fn skeleton(spec_path: &str, out_path: &str) {
    let spec_bytes = match std::fs::read(spec_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("rusty skeleton: read {spec_path}: {e}");
            std::process::exit(1);
        }
    };
    let v: serde_json::Value = match serde_json::from_slice(&spec_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("rusty skeleton: parse spec {spec_path}: {e}");
            std::process::exit(1);
        }
    };
    let arch = v
        .get("arch")
        .and_then(|x| x.as_str())
        .unwrap_or("gasv")
        .to_string();
    let seed = v.get("seed").and_then(|x| x.as_u64()).unwrap_or(0);
    let alignment = v.get("alignment").and_then(|x| x.as_u64()).unwrap_or(32);
    let context_length = v
        .get("context_length")
        .and_then(|x| x.as_u64())
        .unwrap_or(4096) as u32;
    let mut kv = Vec::new();
    if let Some(arr) = v.get("kv").and_then(|x| x.as_array()) {
        for item in arr {
            let k = item.get("key").and_then(|x| x.as_str()).unwrap_or("");
            if k.is_empty() {
                continue;
            }
            let val = match item.get("value") {
                Some(serde_json::Value::String(s)) => rusty_engine::gguf::Value::Str(s.clone()),
                Some(serde_json::Value::Bool(b)) => rusty_engine::gguf::Value::Bool(*b),
                Some(serde_json::Value::Number(n)) => {
                    if let Some(u) = n.as_u64() {
                        rusty_engine::gguf::Value::U64(u)
                    } else if let Some(i) = n.as_i64() {
                        rusty_engine::gguf::Value::I64(i)
                    } else if let Some(f) = n.as_f64() {
                        rusty_engine::gguf::Value::F64(f)
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            kv.push((k.to_string(), val));
        }
    }
    let mut tensors = Vec::new();
    if let Some(arr) = v.get("tensors").and_then(|x| x.as_array()) {
        for item in arr {
            let name = item
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let ggml_type = item.get("ggml_type").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let dims: Vec<usize> = item
                .get("dims")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|u| u as usize))
                        .collect()
                })
                .unwrap_or_default();
            if name.is_empty() || dims.is_empty() {
                eprintln!("rusty skeleton: invalid tensor entry in spec");
                std::process::exit(2);
            }
            tensors.push(rusty_engine::gguf::SkeletonTensorSpec {
                name,
                dims,
                ggml_type,
            });
        }
    }
    let spec = rusty_engine::gguf::SkeletonSpec {
        arch,
        seed,
        alignment,
        context_length,
        tensors,
        kv,
    };
    match rusty_engine::gguf::build_skeleton(&spec) {
        Ok((bytes, report)) => {
            if let Err(e) = std::fs::write(out_path, &bytes) {
                eprintln!("rusty skeleton: write {out_path}: {e}");
                std::process::exit(1);
            }
            let sha256 = rusty_engine::patch::sha256_hex(&bytes);
            println!(
                "{}",
                serde_json::json!({
                    "schema": "gasv.skeleton-build.v1",
                    "spec": spec_path,
                    "out": out_path,
                    "bytes": report.bytes_written,
                    "n_tensors": report.n_tensors,
                    "n_kv": report.n_kv,
                    "seed": spec.seed,
                    "arch": spec.arch,
                    "sha256": sha256,
                })
            );
        }
        Err(e) => {
            eprintln!("rusty skeleton: {e}");
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
/// Dump one tensor as raw little-endian f32 bytes (LE copy for F32, dequant for
/// quantized). Used to stage MiniCPM weights for the GSV-WGPU mesh (MC-2): peers
/// verify chunk sha256 and compute matvecs on-device without any synthetic weights.
fn dump_f32(path: &str, name: &str, out: &str) {
    let model = match unsafe { rusty_engine::loader::MmapModel::open(Path::new(path)) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rusty dump-f32 {path}: {e}");
            std::process::exit(1);
        }
    };
    let entry = match model.header().tensors.iter().find(|t| t.name == name) {
        Some(t) => t,
        None => {
            eprintln!(
                "rusty dump-f32: tensor '{name}' not found in {} tensors",
                model.header().tensors.len()
            );
            std::process::exit(1);
        }
    };
    let numel = entry.numel();
    let floats: Vec<f32> = if entry.ggml_type == 0 {
        match model.tensor_bytes(name) {
            Some(b) => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            None => {
                eprintln!("rusty dump-f32: tensor '{name}' F32 bounds-check failed");
                std::process::exit(1);
            }
        }
    } else {
        match model.dequant(name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("rusty dump-f32: dequant '{name}': {e}");
                std::process::exit(1);
            }
        }
    };
    let mut bytes = Vec::with_capacity(floats.len() * 4);
    for v in &floats {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(out, &bytes) {
        eprintln!("rusty dump-f32: write '{out}': {e}");
        std::process::exit(1);
    }
    println!(
        "'{name}' dims={:?} ggml={} numel={numel} -> '{out}' ({} bytes LE f32)",
        entry.dims,
        entry.ggml_type,
        bytes.len()
    );
}

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

/// `rusty tokenize -m <model.gguf> -p "text"` — dump BPE token ids with no weight load
/// (the `PH-1305` pin pair against llama.cpp's `llama-tokenize`).
fn tokenize_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next(),
            "-p" | "--prompt" => prompt = args.next(),
            other => {
                eprintln!("rusty tokenize: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("usage: rusty tokenize -m <model.gguf> -p \"text\"");
            std::process::exit(2);
        }
    };
    let text = match prompt {
        Some(t) => t,
        None => {
            eprintln!("rusty tokenize: -p <text> is required");
            std::process::exit(2);
        }
    };
    let file = match rusty_engine::gguf::parse_file(
        Path::new(&path),
        rusty_engine::gguf::INSPECT_WINDOW,
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("rusty tokenize {path}: {e}");
            std::process::exit(1);
        }
    };
    let meta = rusty_engine::tokenizer::TokenizerMeta::from_file(&file);
    let bpe = rusty_engine::tokenizer::Bpe::from_meta(meta);
    let ids = match bpe.encode(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("rusty tokenize: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "{}",
        ids.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    std::process::exit(0);
}

/// `rusty run -m <model.gguf> (-p "text" | -t "0,1,2") [--max-tokens N]` — dense greedy
/// generation. `-p` runs the BPE text pipeline (PH-306b); `-t` uses numeric token ids (PH-306).
/// JSON-escape a string for embedding in a hand-rolled JSON object.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// `rusty audit -m <model.gguf> -p "text"` — ActivationAudit stats dump (ECO-9.2 phase 9.3a).
/// Runs the prompt's prefill and emits one JSON object: mean/max per layer of the
/// residual-stream RMS (block entry) and of the block's delta.
fn audit_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut model_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next(),
            "-p" | "--prompt" => prompt = args.next(),
            other => {
                eprintln!("rusty audit: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let (path, text) = match (model_path, prompt) {
        (Some(p), Some(t)) if !p.is_empty() && !t.is_empty() => (p, t),
        _ => {
            eprintln!("usage: rusty audit -m <model.gguf> -p \"text\"");
            std::process::exit(2);
        }
    };
    let model = match unsafe { rusty_engine::loader::MmapModel::open(Path::new(&path)) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rusty audit {path}: {e}");
            std::process::exit(1);
        }
    };
    let mut tr = match rusty_engine::textgen::TextRunner35::from_model(&model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("rusty audit {path}: {e}");
            std::process::exit(1);
        }
    };
    let audit = match tr.audit(&text) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("rusty audit {text}: {e}");
            std::process::exit(1);
        }
    };
    use std::collections::BTreeMap;
    #[derive(Default)]
    struct Agg {
        kind: &'static str,
        sum_entry: f64,
        sum_delta: f64,
        max_delta: f32,
        n: usize,
    }
    let mut by_layer: BTreeMap<usize, Agg> = BTreeMap::new();
    for e in &audit {
        let a = by_layer.entry(e.layer).or_insert_with(|| Agg {
            kind: e.kind,
            ..Agg::default()
        });
        a.sum_entry += e.entry_rms as f64;
        a.sum_delta += e.delta_rms as f64;
        a.max_delta = a.max_delta.max(e.delta_rms);
        a.n += 1;
    }
    let mut out = format!(
        "{{\"model\":\"{}\",\"prompt\":\"{}\",\"prompt_tokens\":{},\"layers\":[",
        json_escape(&path),
        json_escape(&text),
        audit.iter().map(|e| e.token_idx).max().map_or(0, |m| m + 1)
    );
    for (i, (layer, a)) in by_layer.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"layer\":{},\"kind\":\"{}\",\"entry_rms_mean\":{:.4},\"delta_rms_mean\":{:.4},\"delta_rms_max\":{:.4}}}",
            layer,
            a.kind,
            a.sum_entry / a.n as f64,
            a.sum_delta / a.n as f64,
            a.max_delta
        ));
    }
    out.push_str("]}");
    println!("{out}");
    std::process::exit(0);
}

/// `rusty pack-mint -m <model.gguf> -t <tensor> [--zero] (-b <block_idx> | --all)`
/// — mint one wsp-pack line for a block or for every block of a tensor.
/// Default mode: identity receipt (same bits, true base_hash) — useful for
/// validating the serve `--patch` pipeline. `--zero`: replacement is an
/// all-zero block (ablation probes for MC-6). Output lines are valid input
/// for `rusty serve --patch`.
fn pack_mint_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut model_path: Option<String> = None;
    let mut tensor: Option<String> = None;
    let mut block_idx: Option<u32> = None;
    let mut zero = false;
    let mut all = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next(),
            "-t" | "--tensor" => tensor = args.next(),
            "-b" | "--block" => {
                block_idx = args.next().and_then(|v| v.parse::<u32>().ok());
            }
            "--zero" => zero = true,
            "--all" => all = true,
            other => {
                eprintln!("rusty pack-mint: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let (path, name) = match (model_path, tensor) {
        (Some(p), Some(t)) if !p.is_empty() && !t.is_empty() => (p, t),
        _ => {
            eprintln!(
                "usage: rusty pack-mint -m <model.gguf> -t <tensor> (-b <idx> | --all) [--zero]"
            );
            std::process::exit(2);
        }
    };
    let model = match unsafe { rusty_engine::loader::MmapModel::open(Path::new(&path)) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("rusty pack-mint {path}: {e}");
            std::process::exit(1);
        }
    };
    let entry = match model.header().tensors.iter().find(|t| t.name == name) {
        Some(e) => e,
        None => {
            eprintln!("rusty pack-mint: tensor '{name}' not found");
            std::process::exit(1);
        }
    };
    let block_bytes = match rusty_engine::quant::block_info(entry.ggml_type) {
        Some(b) => b.bytes,
        None => {
            eprintln!(
                "rusty pack-mint: tensor '{name}' ggml_type {} unsupported",
                entry.ggml_type
            );
            std::process::exit(1);
        }
    };
    let bytes = match model.tensor_bytes(&name) {
        Some(b) => b,
        None => {
            eprintln!("rusty pack-mint: tensor '{name}' bytes out of bounds");
            std::process::exit(1);
        }
    };
    let total = bytes.len() / block_bytes;
    let emit = |blk: usize| {
        let start = blk * block_bytes;
        let block = &bytes[start..start + block_bytes];
        let base = rusty_engine::patch::sha256_hex(block);
        let bits: String = if zero {
            "00".repeat(block_bytes)
        } else {
            block.iter().map(|b| format!("{b:02x}")).collect()
        };
        println!("{name} {blk} {bits} {base}");
    };
    if all {
        if block_idx.is_some() {
            eprintln!("rusty pack-mint: -b/--block and --all are mutually exclusive");
            std::process::exit(2);
        }
        for blk in 0..total {
            emit(blk);
        }
        std::process::exit(0);
    }
    let blk = match block_idx {
        Some(b) => b as usize,
        None => {
            eprintln!(
                "usage: rusty pack-mint -m <model.gguf> -t <tensor> (-b <idx> | --all) [--zero]"
            );
            std::process::exit(2);
        }
    };
    if blk >= total {
        eprintln!("rusty pack-mint: block {blk} out of range (total {total})");
        std::process::exit(1);
    }
    emit(blk);
    std::process::exit(0);
}

fn run<A: Iterator<Item = String>>(mut args: A) -> ! {
    let mut model_path: Option<String> = None;
    let mut tokens: Vec<u32> = Vec::new();
    let mut prompt: Option<String> = None;
    let mut max_new = 8usize;
    let mut top_k = 0usize;
    let mut vision_prefix: Option<String> = None;
    let mut vision_at: Option<usize> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next(),
            "--vision-prefix" => vision_prefix = args.next(),
            "--vision-at" => {
                vision_at = args.next().and_then(|v| v.parse::<usize>().ok());
                if vision_at.is_none() {
                    eprintln!("rusty run: bad --vision-at");
                    std::process::exit(2);
                }
            }
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
            "--top-k" => {
                top_k = args.next().map_or(5, |v| {
                    v.parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("rusty run: bad --top-k");
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
    // Text pipeline (PH-306b): needs tokenizer.ggml.* meta. PH-1304: `qwen35` arches
    // route to the hybrid runner (delta-state GDN + gated full attention).
    if let Some(text) = prompt {
        let cfg = rusty_engine::model::ModelConfig::from_file(model.header()).ok();
        let q35 = cfg.as_ref().map(|c| c.arch == "qwen35").unwrap_or(false);
        if q35 {
            let mut tr = match rusty_engine::textgen::TextRunner35::from_model(&model) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("rusty run {path}: {e}");
                    std::process::exit(1);
                }
            };
            if let Some(vp) = vision_prefix.as_deref() {
                let d_model = cfg.as_ref().map(|c| c.d_model).unwrap_or(0);
                let vs = match rusty_engine::textgen::load_vision_prefix(vp, d_model) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("rusty run {path}: {e}");
                        std::process::exit(2);
                    }
                };
                if let Err(e) = tr.set_vision_prefix(vs) {
                    eprintln!("rusty run {path}: {e}");
                    std::process::exit(2);
                }
                if let Some(at) = vision_at {
                    tr.set_vision_at(at);
                }
            }
            if top_k > 0 {
                let steps = match tr.trace_text(&text, max_new, top_k) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("rusty run {path}: {e}");
                        std::process::exit(1);
                    }
                };
                for (i, s) in steps.iter().enumerate() {
                    let top: Vec<String> =
                        s.top.iter().map(|(t, l)| format!("{t}:{l:.6}")).collect();
                    println!("{i}\t{}\t{}", s.token, top.join(" "));
                }
                std::process::exit(0);
            }
            match tr.run_text(&text, max_new) {
                Ok(got) => println!("[{}] {}", tr.runner.n_layers, got),
                Err(e) => {
                    eprintln!("rusty run {path}: {e}");
                    std::process::exit(1);
                }
            }
            std::process::exit(0);
        }
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
    if cfg.arch == "qwen35" {
        let runner = match rusty_engine::runner35::Q35Runner::from_model(&model, &cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rusty run {path}: runner35: {e}");
                std::process::exit(1);
            }
        };
        let mut dec = rusty_engine::runner35::Q35Decoder::new(&runner);
        let out = dec.generate_incremental(&tokens, max_new);
        let ids: Vec<String> = out.iter().map(|t| t.to_string()).collect();
        println!(
            "[{}] {} layers (hybrid) — prompt {:?} -> {}",
            cfg.arch,
            cfg.n_layers,
            tokens,
            ids.join(",")
        );
        std::process::exit(0);
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
/// Stage a patched copy of `src_model` using the wsp pack at `pack_file`.
/// Writes `patched_<pid>.gguf` next to the current executable (inside the
/// repo's target dir), returns its path. Fail-closed before any bytes land.
fn stage_patch_from_file(
    src_model: &std::path::Path,
    pack_file: &std::path::Path,
) -> std::path::PathBuf {
    let data = std::fs::read_to_string(pack_file).unwrap_or_else(|e| {
        eprintln!("rusty serve: read patch pack {pack_file:?}: {e}");
        std::process::exit(1);
    });
    let pack = rusty_engine::patch::parse_wsp(&data).unwrap_or_else(|e| {
        eprintln!("rusty serve: parse patch pack {pack_file:?}: {e}");
        std::process::exit(1);
    });
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("rusty serve: current_exe: {e}");
        std::process::exit(1);
    });
    let dir = exe.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| {
        eprintln!("rusty serve: cannot resolve exe dir");
        std::process::exit(1);
    });
    let dst = dir.join(format!("patched_{}.gguf", std::process::id()));
    match rusty_engine::patch::stage_patch(src_model, &dst, &pack) {
        Ok(n) => {
            eprintln!(
                "rusty serve: staged patched model {dst:?} ({n} block{} patched)",
                if n == 1 { "" } else { "s" }
            );
            dst
        }
        Err(e) => {
            eprintln!("rusty serve: patch staging failed: {e}");
            let _ = std::fs::remove_file(&dst);
            std::process::exit(1);
        }
    }
}

/// Band 233 (PH-507): LAN-first — binds `0.0.0.0` by default like `llama_serve`,
/// so the GSV hub and Wi-Fi/edge peers address it by the machine's local address;
/// `--host 127.0.0.1` keeps the old loopback-only posture. Serves one connection
/// per request.
fn serve_cmd<A: Iterator<Item = String>>(mut args: A) -> ! {
    // EN-8: default moved 8090 → 8092 — loopback :8090 permanently belongs to rebook
    // (RB-5 axum `rust_book view`); the dense lane must never fight it again.
    let mut port = 8092u16;
    let mut host = "0.0.0.0".to_string();
    let mut model_path: Option<String> = None;
    let mut patch_file: Option<String> = None;
    let mut vision_prefix: Option<String> = None;
    let mut vision_at: Option<usize> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--port" | "-p" => {
                port = args.next().map_or(8092, |v| {
                    v.parse::<u16>().unwrap_or_else(|_| {
                        eprintln!("rusty serve: bad --port");
                        std::process::exit(2)
                    })
                })
            }
            "--host" => host = args.next().unwrap_or_else(|| "0.0.0.0".to_string()),
            "--model" | "-m" => model_path = args.next(),
            "--patch" => patch_file = args.next(),
            "--vision-prefix" => vision_prefix = args.next(),
            "--vision-at" => {
                vision_at = args.next().and_then(|v| v.parse::<usize>().ok());
                if vision_at.is_none() {
                    eprintln!("rusty serve: bad --vision-at");
                    std::process::exit(2);
                }
            }
            other => {
                eprintln!("rusty serve: unknown arg '{other}'");
                std::process::exit(2);
            }
        }
    }
    let addr = format!("{host}:{port}");
    rusty_engine::log::log_line(
        "serve",
        &format!(
            "bind http://{addr} model={}",
            model_path.as_deref().unwrap_or("<stub>")
        ),
    );
    println!(
        "{} — serving on http://{addr} (Ctrl-C to stop)",
        rusty_engine::banner()
    );
    // With a model, wire real BPE→generate→decode (PH-501b); else the stub route.
    // PH-1306: `qwen35` arches bind the hybrid [`TextRunner35`] on the same serve wire —
    // MiniCPM-V 4.7-1B runs OpenAI-shaped chat through one `rusty serve` command.
    let staged_path: Option<std::path::PathBuf> = match (&model_path, &patch_file) {
        (Some(mp), Some(pf)) => Some(stage_patch_from_file(
            std::path::Path::new(mp),
            std::path::Path::new(pf),
        )),
        _ => None,
    };
    let model_path = match &staged_path {
        Some(dst) => Some(dst.to_string_lossy().into_owned()),
        None => model_path,
    };
    let result = match &model_path {
        Some(p) => match unsafe { rusty_engine::loader::MmapModel::open(Path::new(p)) } {
            Ok(m) => {
                let cfg = rusty_engine::model::ModelConfig::from_file(m.header()).ok();
                let hybrid = cfg.as_ref().map(|c| c.arch == "qwen35").unwrap_or(false);
                if hybrid {
                    match rusty_engine::textgen::TextRunner35::from_model(&m) {
                        Ok(mut tr) => {
                            if let Some(vp) = vision_prefix.as_deref() {
                                let d_model = cfg.as_ref().map(|c| c.d_model).unwrap_or(0);
                                let vs =
                                    match rusty_engine::textgen::load_vision_prefix(vp, d_model) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            eprintln!("rusty serve: {e}");
                                            std::process::exit(2);
                                        }
                                    };
                                if let Err(e) = tr.set_vision_prefix(vs) {
                                    eprintln!("rusty serve: {e}");
                                    std::process::exit(2);
                                }
                                if let Some(at) = vision_at {
                                    tr.set_vision_at(at);
                                }
                                rusty_engine::log::log_line(
                                    "model",
                                    &format!("vision prefix loaded: {vp}"),
                                );
                            }
                            rusty_engine::log::log_line(
                                "model",
                                &format!("{p} qwen35 hybrid {} layers", tr.runner.n_layers),
                            );
                            println!("  model: {p} ({} layers, hybrid)", tr.runner.n_layers);
                            rusty_engine::serve::serve_with_model(&addr, &tr)
                        }
                        Err(e) => {
                            rusty_engine::log::log_line(
                                "model",
                                &format!("{p} hybrid load failed: {e}"),
                            );
                            eprintln!("rusty serve: model load: {e}");
                            std::process::exit(1);
                        }
                    }
                } else {
                    match rusty_engine::textgen::TextRunner::from_model(&m) {
                        Ok(tr) => {
                            rusty_engine::log::log_line(
                                "model",
                                &format!("{p} dense {} layers", tr.runner.n_layers),
                            );
                            println!("  model: {p} ({} layers)", tr.runner.n_layers);
                            rusty_engine::serve::serve_with_model(&addr, &tr)
                        }
                        Err(e) => {
                            rusty_engine::log::log_line("model", &format!("{p} load failed: {e}"));
                            eprintln!("rusty serve: model load: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }
            Err(e) => {
                rusty_engine::log::log_line("model", &format!("{p} open failed: {e}"));
                eprintln!("rusty serve: open {p}: {e}");
                std::process::exit(1);
            }
        },
        None => rusty_engine::serve::serve(&addr),
    };
    if let Err(e) = result {
        rusty_engine::log::log_line("serve", &format!("serve error on {addr}: {e}"));
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

fn transplant_mirror(skel_path: &str, tensors_json: &str, chunks_dir: &str, out_path: &str) {
    use rusty_engine::gguf::{
        parse_file, read_tensor_bytes, GgufWriter, TensorData, TensorWrite, INSPECT_WINDOW,
    };
    let skel_path_p = std::path::Path::new(skel_path);
    let skel = match parse_file(skel_path_p, INSPECT_WINDOW) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rusty transplant-mirror: open skeleton {skel_path}: {e}");
            std::process::exit(1);
        }
    };
    let tj_bytes = match std::fs::read(tensors_json) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("rusty transplant-mirror: read {tensors_json}: {e}");
            std::process::exit(1);
        }
    };
    let tj: serde_json::Value = match serde_json::from_slice(&tj_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("rusty transplant-mirror: parse {tensors_json}: {e}");
            std::process::exit(1);
        }
    };
    let arr = match tj.as_array() {
        Some(a) => a,
        None => {
            eprintln!("rusty transplant-mirror: {tensors_json} must be a JSON array");
            std::process::exit(2);
        }
    };
    let mut map: std::collections::BTreeMap<String, (Vec<usize>, u32, usize, Vec<String>)> =
        std::collections::BTreeMap::new();
    for item in arr {
        let name = item.get("tensor").and_then(|x| x.as_str()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let dims: Vec<usize> = item
            .get("dims")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        let numel: usize = item.get("numel").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let ggml = 0u32; // only f32 chunks supported in this path
        let chunks: Vec<String> = item
            .get("chunks")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        map.insert(name.to_string(), (dims, ggml, numel, chunks));
    }

    let mut w = GgufWriter::new_v3().with_alignment(skel.alignment());
    for (k, v) in &skel.metadata {
        w.add_kv(k.clone(), v.clone());
    }
    let mut replaced = 0usize;
    let mut kept = 0usize;
    let mut report: Vec<serde_json::Value> = Vec::new();
    for t in &skel.tensors {
        let mut did = false;
        if let Some((dims, ggml, numel, chunks)) = map.get(&t.name) {
            if &t.dims == dims && t.ggml_type == *ggml && !chunks.is_empty() {
                let mut bytes: Vec<u8> = Vec::with_capacity(numel * 4);
                let mut ok = true;
                for ch in chunks {
                    let p = std::path::Path::new(chunks_dir).join(format!("{ch}.bin"));
                    match std::fs::read(&p) {
                        Ok(b) => bytes.extend_from_slice(&b),
                        Err(e) => {
                            eprintln!("rusty transplant-mirror: read chunk {p:?}: {e}");
                            ok = false;
                            break;
                        }
                    }
                }
                let expected = numel * 4;
                if ok && bytes.len() == expected {
                    w.add_tensor(TensorWrite {
                        name: t.name.clone(),
                        dims: t.dims.clone(),
                        ggml_type: t.ggml_type,
                        data: TensorData::Raw(bytes.clone()),
                    });
                    report.push(serde_json::json!({"tensor":t.name,"status":"transplanted","bytes":bytes.len(),"chunks":chunks.len()}));
                    replaced += 1;
                    did = true;
                } else if ok {
                    eprintln!(
                        "rusty transplant-mirror: {} byte mismatch got {} expected {}",
                        t.name,
                        bytes.len(),
                        expected
                    );
                    std::process::exit(1);
                }
            }
        }
        if !did {
            // keep skeleton tensor bytes
            let bytes = match read_tensor_bytes(skel_path_p, &skel, &t.name) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "rusty transplant-mirror: read skeleton tensor {}: {e}",
                        t.name
                    );
                    std::process::exit(1);
                }
            };
            w.add_tensor(TensorWrite {
                name: t.name.clone(),
                dims: t.dims.clone(),
                ggml_type: t.ggml_type,
                data: TensorData::Raw(bytes),
            });
            report.push(serde_json::json!({"tensor":t.name,"status":"kept_skeleton"}));
            kept += 1;
        }
    }
    match w.write_to_file(std::path::Path::new(out_path)) {
        Ok(wr) => {
            let out = serde_json::json!({
                "schema":"gasv.transplant-mirror.v1",
                "out":out_path,
                "replaced":replaced,
                "kept_skeleton":kept,
                "n_tensors":wr.n_tensors,
                "bytes_written":wr.bytes_written,
                "report":report
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&out).unwrap_or_else(|_| "{}".into())
            );
        }
        Err(e) => {
            eprintln!("rusty transplant-mirror: write {out_path}: {e}");
            std::process::exit(1);
        }
    }
}

fn artifact_report(
    path: &str,
    kind: &str,
    runnable: bool,
    source: &str,
    coverage: &str,
    transplant_sha: &str,
) {
    use rusty_engine::gguf::{parse_file, INSPECT_WINDOW};
    let p = std::path::Path::new(path);
    let g = match parse_file(p, INSPECT_WINDOW) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("rusty artifact-report: open {path}: {e}");
            std::process::exit(1);
        }
    };
    let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let sha256 = std::fs::read(p)
        .ok()
        .map(|b| rusty_engine::patch::sha256_hex(&b))
        .unwrap_or_default();
    let arch = g
        .get("general.architecture")
        .and_then(|v| match v {
            rusty_engine::gguf::Value::Str(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("unknown");
    let seed = g
        .get("gsv.seed")
        .and_then(|v| match v {
            rusty_engine::gguf::Value::U64(x) => Some(*x),
            rusty_engine::gguf::Value::U32(x) => Some(*x as u64),
            _ => None,
        })
        .unwrap_or(0);
    let ctx = g
        .get("gsv.context_length")
        .and_then(|v| match v {
            rusty_engine::gguf::Value::U32(x) => Some(*x),
            rusty_engine::gguf::Value::U64(x) => Some(*x as u32),
            _ => None,
        })
        .unwrap_or(0);
    let mut dtypes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for t in &g.tensors {
        dtypes.insert(format!("{}", t.ggml_type));
    }
    let report = build_artifact_report(
        path,
        kind,
        runnable,
        arch,
        seed,
        ctx,
        g.tensors.len(),
        bytes,
        &sha256,
        dtypes.into_iter().collect::<Vec<_>>(),
        source,
        coverage,
        transplant_sha,
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
    );
}

/// GASV-F1: pure artifact-contract builder. Kept separate from file I/O so
/// the schema (especially the honesty fields) can be unit-tested.
#[allow(clippy::too_many_arguments)]
fn build_artifact_report(
    path: &str,
    kind: &str,
    runnable: bool,
    arch: &str,
    seed: u64,
    ctx: u32,
    tensor_count: usize,
    bytes: u64,
    sha256: &str,
    ggml_types: Vec<String>,
    source: &str,
    coverage: &str,
    transplant_sha: &str,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "gasv.artifact-manifest.v1",
        "kind": kind,
        "runnable": runnable,
        "artifact": path,
        "arch": arch,
        "seed": seed,
        "context_length": ctx,
        "tensor_count": tensor_count,
        "bytes": bytes,
        "sha256": sha256,
        "ggml_types": ggml_types,
        "source": source,
        "coverage": coverage,
        "transplant_report_sha256": transplant_sha,
        "nanojev_role": "eval-oracle-only",
        "no_quality_parity_claim": true,
    })
}

fn quantize_export(inp: &str, out_path: &str, quant: &str, report_path: Option<&str>) {
    use rusty_engine::gguf::{
        parse_file, read_tensor_bytes, GgufWriter, TensorData, TensorWrite, INSPECT_WINDOW,
    };
    use rusty_engine::quant::{bytes_for_numel, dequant, quantize};
    let (target_type, quant_policy): (u32, &str) = match quant {
        "q4_0" | "Q4_0" => (2, "q4_0"),
        "q8_0" | "Q8_0" => (8, "q8_0"),
        "q4_k" | "Q4_K" => (12, "q4_k"),
        // Q4_K_M without imatrix: fallback to Q4_K for all tensors.
        // Full Q4_K_M mixed policy (Q6_K for selected tensors) requires Q6_K encoder + imatrix.
        "q4_k_m" | "Q4_K_M" => (12, "q4_k_m-no-imatrix-fallback=q4_k"),
        _ => {
            eprintln!(
                "rusty quantize-export: unsupported quant '{quant}' (supported: q4_0, q8_0, q4_k, q4_k_m[fallback])"
            );
            std::process::exit(2);
        }
    };
    let inp_p = std::path::Path::new(inp);
    let src = match parse_file(inp_p, INSPECT_WINDOW) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rusty quantize-export: open {inp}: {e}");
            std::process::exit(1);
        }
    };
    let mut w = GgufWriter::new_v3().with_alignment(src.alignment());
    for (k, v) in &src.metadata {
        w.add_kv(k.clone(), v.clone());
    }
    let mut report_rows: Vec<serde_json::Value> = Vec::new();
    let mut total_src_bytes: usize = 0;
    let mut total_out_bytes: usize = 0;
    for t in &src.tensors {
        let raw = match read_tensor_bytes(inp_p, &src, &t.name) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("rusty quantize-export: read tensor {}: {e}", t.name);
                std::process::exit(1);
            }
        };
        total_src_bytes += raw.len();
        let numel: usize = t.dims.iter().product::<usize>();
        // GASV-G32: projector/vision GGUFs store weights as F16 (1) or BF16 (30),
        // not F32. `quant::dequant` already widens both losslessly to f32, so
        // accept them here instead of refusing to produce a 4-bit vision block.
        if !matches!(t.ggml_type, 0 | 1 | 30) {
            eprintln!(
                "rusty quantize-export: tensor {} is ggml_type {} (supported sources: F32=0, F16=1, BF16=30)",
                t.name, t.ggml_type
            );
            std::process::exit(1);
        }
        let f32s = match dequant(t.ggml_type, &raw) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("rusty quantize-export: dequant {}: {}", t.name, e.message);
                std::process::exit(1);
            }
        };
        if f32s.len() != numel {
            eprintln!(
                "rusty quantize-export: tensor {} dequant gave {} values, expected {}",
                t.name,
                f32s.len(),
                numel
            );
            std::process::exit(1);
        }
        // Q4_K super-block is 256; small tensors fall back to Q8_0 (or keep F32).
        let (actual_type, qbytes) = if target_type == 12 && numel % 256 != 0 {
            let fb = if numel % 32 == 0 { 8u32 } else { 0u32 };
            let b = if fb == 0 {
                // keep F32 raw
                let mut raw_f32 = Vec::with_capacity(numel * 4);
                for &v in &f32s {
                    raw_f32.extend_from_slice(&v.to_le_bytes());
                }
                raw_f32
            } else {
                match quantize(fb, &f32s) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "rusty quantize-export: fallback quantize {}: {}",
                            t.name, e.message
                        );
                        std::process::exit(1);
                    }
                }
            };
            (fb, b)
        } else {
            match quantize(target_type, &f32s) {
                Ok(b) => (target_type, b),
                Err(e) => {
                    eprintln!("rusty quantize-export: quantize {}: {}", t.name, e.message);
                    std::process::exit(1);
                }
            }
        };
        let expected = match bytes_for_numel(actual_type, numel) {
            Some(n) => n,
            None => {
                eprintln!(
                    "rusty quantize-export: cannot size {} numel={numel} type={actual_type}",
                    t.name
                );
                std::process::exit(1);
            }
        };
        if qbytes.len() != expected {
            eprintln!(
                "rusty quantize-export: {} quantized bytes {} != expected {}",
                t.name,
                qbytes.len(),
                expected
            );
            std::process::exit(1);
        }
        let back = match dequant(actual_type, &qbytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "rusty quantize-export: dequant check {}: {}",
                    t.name, e.message
                );
                std::process::exit(1);
            }
        };
        let mut max_abs: f32 = 0.0;
        let mut sum_abs: f64 = 0.0;
        let ncmp = f32s.len().min(back.len());
        for i in 0..ncmp {
            let d = (f32s[i] - back[i]).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
        }
        let mean_abs = if ncmp > 0 { sum_abs / ncmp as f64 } else { 0.0 };
        total_out_bytes += qbytes.len();
        report_rows.push(serde_json::json!({
            "tensor": t.name,
            "dims": t.dims,
            "numel": numel,
            "src_ggml_type": 0,
            "dst_ggml_type": actual_type,
            "requested_ggml_type": target_type,
            "fallback": actual_type != target_type,
            "src_bytes": raw.len(),
            "dst_bytes": qbytes.len(),
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
        }));
        w.add_tensor(TensorWrite {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: actual_type,
            data: TensorData::Raw(qbytes),
        });
    }
    match w.write_to_file(std::path::Path::new(out_path)) {
        Ok(wr) => {
            let out_sha = std::fs::read(out_path)
                .ok()
                .map(|b| rusty_engine::patch::sha256_hex(&b))
                .unwrap_or_default();
            let report = serde_json::json!({
                "schema": "gasv.quant-export-report.v1",
                "input": inp,
                "output": out_path,
                "quant": quant,
                "quant_policy": quant_policy,
                "target_ggml_type": target_type,
                "n_tensors": wr.n_tensors,
                "bytes_written": wr.bytes_written,
                "sha256": out_sha,
                "total_src_bytes": total_src_bytes,
                "total_out_bytes": total_out_bytes,
                "compression_ratio": if total_out_bytes > 0 { total_src_bytes as f64 / total_out_bytes as f64 } else { 0.0 },
                "tensors": report_rows,
            });
            if let Some(rp) = report_path {
                if let Err(e) = std::fs::write(
                    rp,
                    serde_json::to_string_pretty(&report).unwrap_or_default(),
                ) {
                    eprintln!("rusty quantize-export: write report {rp}: {e}");
                    std::process::exit(1);
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
            );
        }
        Err(e) => {
            eprintln!("rusty quantize-export: write {out_path}: {e}");
            std::process::exit(1);
        }
    }
}

/// `rusty artifact-chunks` — GASV-G20: emit a content-addressed artifact-source
/// manifest for one GGUF tensor: slice its raw bytes into <=max pieces, hash
/// each piece with SHA-256 to form the chunk key, and report (key, offset, len)
/// so GSV can range-serve chunks from the artifact without duplicating them.
fn artifact_chunks(inp: &str, tensor_name: &str, max: usize) {
    use rusty_engine::gguf::{parse_file, INSPECT_WINDOW};
    use std::io::{Read, Seek, SeekFrom};
    let inp_p = std::path::Path::new(inp);
    let header = match parse_file(inp_p, INSPECT_WINDOW) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("rusty artifact-chunks: open {inp}: {e}");
            std::process::exit(1);
        }
    };
    let (start, end) = match header.tensor_data_range(tensor_name) {
        Some(r) => r,
        None => {
            eprintln!("rusty artifact-chunks: tensor `{tensor_name}` not found / unsized");
            std::process::exit(1);
        }
    };
    let entry = header
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .expect("entry exists");
    let total = end - start;
    let max = max.max(1);
    let mut f = match std::fs::File::open(inp_p) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("rusty artifact-chunks: read {inp}: {e}");
            std::process::exit(1);
        }
    };
    let mut chunks = Vec::new();
    let mut off = start;
    while off < end {
        let len = max.min(end - off);
        let mut buf = vec![0u8; len];
        if let Err(e) = f
            .seek(SeekFrom::Start(off as u64))
            .and_then(|_| f.read_exact(&mut buf))
        {
            eprintln!(
                "rusty artifact-chunks: range read {off}..{}: {e}",
                off + len
            );
            std::process::exit(1);
        }
        let key = rusty_engine::patch::sha256_hex(&buf);
        chunks.push(serde_json::json!({
            "key": key,
            "offset": off as u64,
            "len": len as u64,
        }));
        off += len;
    }
    let out = serde_json::json!({
        "schema": "gasv.artifact-chunks.v1",
        "artifact": inp,
        "tensor": tensor_name,
        "dims": entry.dims,
        "ggml_type": entry.ggml_type,
        "bytes": total,
        "max_chunk_bytes": max,
        "chunks": chunks,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
}

fn assemble_copy(
    inp: &str,
    out_path: &str,
    report_path: Option<&str>,
    add_kv: &[(String, String)],
) {
    use rusty_engine::gguf::{
        parse_file, read_tensor_bytes, GgufWriter, TensorData, TensorWrite, Value, INSPECT_WINDOW,
    };
    let inp_p = std::path::Path::new(inp);
    let src = match parse_file(inp_p, INSPECT_WINDOW) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rusty assemble-copy: open {inp}: {e}");
            std::process::exit(1);
        }
    };
    let mut w = GgufWriter::new_v3().with_alignment(src.alignment());
    for (k, v) in &src.metadata {
        // Skip existing GSV provenance keys we'll replace
        if k.starts_with("gsv.assemble.") {
            continue;
        }
        w.add_kv(k.clone(), v.clone());
    }
    for (k, v) in add_kv {
        w.add_kv(k.clone(), Value::Str(v.clone()));
    }
    w.add_kv(
        "gsv.assemble.schema",
        Value::Str("gasv.assemble-copy.v1".into()),
    );
    w.add_kv("gsv.assemble.source", Value::Str(inp.to_string()));
    w.add_kv(
        "gsv.assemble.mode",
        Value::Str("direct-copy-raw-bytes".into()),
    );

    let mut tensor_rows: Vec<serde_json::Value> = Vec::new();
    for t in &src.tensors {
        let numel: usize = t.dims.iter().product();
        let byte_len = match rusty_engine::quant::bytes_for_numel(t.ggml_type, numel) {
            Some(n) => n,
            None => {
                eprintln!(
                    "rusty assemble-copy: cannot size tensor {} type {} dims {:?}",
                    t.name, t.ggml_type, t.dims
                );
                std::process::exit(1);
            }
        };
        tensor_rows.push(serde_json::json!({
            "name": t.name,
            "dims": t.dims,
            "ggml_type": t.ggml_type,
            "numel": numel,
            "bytes": byte_len,
        }));
        // data field ignored by write_to_file_streaming, but fill with empty placeholder
        w.add_tensor(TensorWrite {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            data: TensorData::Raw(Vec::new()),
        });
    }
    let out_p = std::path::Path::new(out_path);
    let inp_p2 = inp_p.to_path_buf();
    let src2 = src.clone();
    let res =
        w.write_to_file_streaming(out_p, move |tw| read_tensor_bytes(&inp_p2, &src2, &tw.name));
    match res {
        Ok(wr) => {
            let out_sha = rusty_engine::patch::sha256_hex_file(out_p).unwrap_or_default();
            let report = serde_json::json!({
                "schema": "gasv.assemble-copy-report.v1",
                "input": inp,
                "output": out_path,
                "n_tensors": wr.n_tensors,
                "n_kv": wr.n_kv,
                "bytes_written": wr.bytes_written,
                "sha256": out_sha,
                "tensors": tensor_rows,
            });
            if let Some(rp) = report_path {
                if let Err(e) = std::fs::write(
                    rp,
                    serde_json::to_string_pretty(&report).unwrap_or_default(),
                ) {
                    eprintln!("rusty assemble-copy: write report {rp}: {e}");
                    std::process::exit(1);
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".into())
            );
        }
        Err(e) => {
            eprintln!("rusty assemble-copy: write {out_path}: {e}");
            std::process::exit(1);
        }
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

#[cfg(test)]
mod tests {
    use super::build_artifact_report;

    /// GASV-F1: the partial shexp transplant contract must always carry the
    /// honesty fields — never a quality-parity claim.
    #[test]
    fn artifact_contract_is_partial_not_runnable() {
        let r = build_artifact_report(
            "partial-shexp.gguf",
            "partial-shexp-pack",
            false,
            "gsv",
            17,
            512,
            121,
            4096,
            "abc123",
            vec!["f32".into()],
            "tensors.json+chunks",
            "blk.0..4 shexp-only",
            "deadbeef",
        );
        assert_eq!(r["schema"], "gasv.artifact-manifest.v1");
        assert_eq!(r["kind"], "partial-shexp-pack");
        assert_eq!(r["runnable"], false);
        assert_eq!(r["arch"], "gsv");
        assert_eq!(r["seed"], 17);
        assert_eq!(r["tensor_count"], 121);
        assert_eq!(r["bytes"], 4096);
        assert_eq!(r["sha256"], "abc123");
        assert_eq!(r["coverage"], "blk.0..4 shexp-only");
        assert_eq!(r["transplant_report_sha256"], "deadbeef");
        assert_eq!(r["nanojev_role"], "eval-oracle-only");
        assert_eq!(r["no_quality_parity_claim"], true);
    }

    /// The builder can describe a full artifact too, but the honesty fields
    /// stay contract constants.
    #[test]
    fn artifact_contract_honesty_fields_are_constants() {
        let r = build_artifact_report(
            "full.gguf",
            "full",
            true,
            "gsv",
            0,
            0,
            1,
            1,
            "x",
            vec![],
            "",
            "",
            "",
        );
        assert_eq!(r["runnable"], true);
        assert_eq!(r["nanojev_role"], "eval-oracle-only");
        assert_eq!(r["no_quality_parity_claim"], true);
    }
}
