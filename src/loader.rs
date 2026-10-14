//! `rusty` model loaders — band B1, `PH-103`.
//!
//! Two ways to get a GGUF from disk, mirroring the llama-rs "staged loader" idea
//! (disk→RAM, optional page-lock, progress) but with **no C++ backend**:
//! - [`MmapModel`] — zero-copy via `memmap2` (the engine's first non-`std` dep).
//! - [`staged_read`] — chunked `Read` into RAM with a `(done, total)` callback, for
//!   hosts without mmap or when the caller wants explicit progress.
//!
//! Tensor *data* dequant is `PH-104`; these loaders only surface the bytes + header.

use crate::gguf::{parse, GgufFile, TensorEntry};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Round `off` up to the next multiple of `alignment` (the GGUF data-start rule).
pub fn aligned_start(off: usize, alignment: u64) -> usize {
    let a = alignment.max(1) as usize;
    off.div_ceil(a) * a
}

/// A model whose bytes live in the OS page cache via mmap (zero-copy tensor access).
pub struct MmapModel {
    path: PathBuf,
    mmap: memmap2::Mmap,
    header: GgufFile,
    data_start: usize,
}

impl MmapModel {
    /// Map `path` and parse its GGUF header.
    ///
    /// # Safety
    /// The caller must guarantee the backing file is not modified (and its directory
    /// entry is not swapped) for the lifetime of the returned map — the standard
    /// `memmap2` contract; mutating it is undefined behaviour.
    pub unsafe fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let header = parse(&mmap[..]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let data_start = aligned_start(header.data_offset, header.alignment());
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            header,
            data_start,
        })
    }

    /// Best-effort `mlock` (pin pages resident); returns whether it succeeded.
    /// `memmap2` only exposes `lock` on Unix; other targets report `false`.
    pub fn lock_pages(&self) -> bool {
        #[cfg(unix)]
        {
            self.mmap.lock().is_ok()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &GgufFile {
        &self.header
    }

    /// Start byte of the aligned tensor-data blob.
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// The tensor-data region (aligned start → EOF); empty if none present.
    pub fn data(&self) -> &[u8] {
        self.mmap.get(self.data_start..).unwrap_or(&[])
    }

    /// Bounds-checked raw block slice `[from, from + len)` within the mapped file.
    pub fn slice(&self, from: usize, len: usize) -> Option<&[u8]> {
        self.mmap.get(from..from.checked_add(len)?)
    }

    /// Total mapped bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Byte range `[data_start + tensor.offset, +nbytes)` of a named tensor, bounds-checked
    /// (the loader's exact guard). `offset` is a byte delta; never divide it by type size.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let e = self.header.tensors.iter().find(|t| t.name == name)?;
        let len = e.nbytes()?;
        let start = self.data_start.checked_add(e.offset as usize)?;
        self.mmap.get(start..start.checked_add(len)?)
    }

    /// Element count of a named tensor.
    pub fn tensor_numel(&self, name: &str) -> Option<usize> {
        self.header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(TensorEntry::numel)
    }

    /// Dequantize a named tensor through the scalar oracle
    /// (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K; other types return `Err`).
    pub fn dequant(&self, name: &str) -> crate::core::Result<Vec<f32>> {
        let e = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| crate::core::EngineError {
                message: format!("tensor '{name}' not found"),
            })?;
        let bytes = self
            .tensor_bytes(name)
            .ok_or_else(|| crate::core::EngineError {
                message: format!("tensor '{name}' bytes out of bounds or type unsized"),
            })?;
        crate::quant::dequant(e.ggml_type, bytes)
    }
}

/// Read `path` into RAM in `chunk`-sized pieces, calling `progress(done, total)` after
/// each chunk (and once with `0` at the start). The "staged" loader for explicit progress.
pub fn staged_read<P: FnMut(usize, usize)>(
    path: &Path,
    chunk: usize,
    mut progress: P,
) -> io::Result<Vec<u8>> {
    let chunk = chunk.max(1);
    let mut file = File::open(path)?;
    let total = file.metadata()?.len() as usize;
    let mut buf = Vec::with_capacity(total.min(64 * 1024 * 1024));
    let mut tmp = vec![0u8; chunk];
    let mut done = 0usize;
    progress(0, total);
    loop {
        let n = file.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        done += n;
        progress(done, total);
    }
    Ok(buf)
}

/// `staged_read` + parse: a fully RAM-resident model (no mmap), reporting no progress.
pub fn load_staged(path: &Path) -> io::Result<(GgufFile, Vec<u8>)> {
    let bytes = staged_read(path, 4 * 1024 * 1024, |_, _| {})?;
    let header = parse(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((header, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF v3 header (1 KV + 1 tensor-info) padded to alignment(32) with a
    /// 16-byte tensor-data blob appended, so `data_start` points at real bytes.
    fn synth_with_data() -> Vec<u8> {
        fn u32le(b: &mut Vec<u8>, x: u32) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fn u64le(b: &mut Vec<u8>, x: u64) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fn s(b: &mut Vec<u8>, v: &str) {
            u64le(b, v.len() as u64);
            b.extend_from_slice(v.as_bytes());
        }
        let mut b = Vec::new();
        b.extend_from_slice(crate::gguf::GGUF_MAGIC);
        u32le(&mut b, 3); // version
        u64le(&mut b, 1); // n_tensors
        u64le(&mut b, 1); // n_kv
        s(&mut b, "general.architecture");
        u32le(&mut b, 8); // Str
        s(&mut b, "qwen2moe");
        s(&mut b, "w"); // tensor name
        u32le(&mut b, 1); // n_dims
        u64le(&mut b, 4); // dim0
        u32le(&mut b, 0); // ggml F32
        u64le(&mut b, 0); // offset
        let aligned = aligned_start(b.len(), 32);
        b.resize(aligned, 0);
        b.extend_from_slice(&[1u8; 16]); // tensor data (4 x f32)
        b
    }

    #[test]
    fn aligned_start_rounds_up() {
        assert_eq!(aligned_start(0, 32), 0);
        assert_eq!(aligned_start(1, 32), 32);
        assert_eq!(aligned_start(32, 32), 32);
        assert_eq!(aligned_start(33, 32), 64);
        assert_eq!(aligned_start(5, 0), 5); // alignment.max(1) guard
    }

    #[test]
    fn mmap_open_parses_header_and_data() {
        let bytes = synth_with_data();
        std::fs::create_dir_all("target").unwrap();
        let path = std::path::Path::new("target/rusty_mmap_fixture.gguf");
        std::fs::write(path, &bytes).unwrap();
        let model = unsafe { MmapModel::open(path) }.expect("mmap open");
        assert_eq!(model.header().version, 3);
        assert_eq!(model.header().n_tensors, 1);
        assert_eq!(model.len(), bytes.len());
        assert_eq!(model.data_start() % 32, 0);
        assert!(model.data_start() <= model.len());
        assert_eq!(model.data().len(), 16);
        assert_eq!(
            model.slice(model.data_start(), 16).map(|s| s.len()),
            Some(16)
        );
        assert!(model.slice(model.data_start(), 999).is_none());
        drop(model);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn staged_read_reports_full_progress() {
        let bytes = synth_with_data();
        std::fs::create_dir_all("target").unwrap();
        let path = std::path::Path::new("target/rusty_staged_fixture.gguf");
        std::fs::write(path, &bytes).unwrap();
        let mut last = (0usize, 0usize);
        let got = staged_read(path, 7, |d, t| {
            last = (d, t);
        })
        .unwrap();
        assert_eq!(got.len(), bytes.len());
        assert_eq!(last.0, last.1); // final report reached total
        let _ = std::fs::remove_file(path);
    }

    /// GGUF `offset` is a **byte** delta from `data_start` (llama.cpp semantics). This
    /// lays out two real-format tensors and checks `tensor_bytes`/`dequant` apply it.
    #[test]
    fn tensor_bytes_applies_byte_offset_on_real_layout() {
        fn u32le(b: &mut Vec<u8>, x: u32) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fn u64le(b: &mut Vec<u8>, x: u64) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        fn tn(b: &mut Vec<u8>, s: &str) {
            u64le(b, s.len() as u64);
            b.extend_from_slice(s.as_bytes());
        }

        let mut h = Vec::new();
        h.extend_from_slice(crate::gguf::GGUF_MAGIC);
        u32le(&mut h, 3);
        u64le(&mut h, 2); // n_tensors
        u64le(&mut h, 0); // n_kv
        tn(&mut h, "norm");
        u32le(&mut h, 1);
        u64le(&mut h, 8); // dims [8]
        u32le(&mut h, 0); // F32
        u64le(&mut h, 0); // offset
        tn(&mut h, "w");
        u32le(&mut h, 1);
        u64le(&mut h, 32); // dims [32]
        u32le(&mut h, 2); // Q4_0
        u64le(&mut h, 32); // offset = PAD(32,32)

        let data_start = aligned_start(h.len(), 32);
        let mut file = h;
        file.resize(data_start, 0);
        for i in 0..8u32 {
            u32le(&mut file, ((i + 1) as f32).to_bits());
        }
        file.extend_from_slice(&[0x00, 0x3C]); // Q4_0 scale 1.0
        file.extend_from_slice(&[0u8; 16]); // qs zero -> -8 each

        std::fs::create_dir_all("target").unwrap();
        let path = std::path::Path::new("target/rusty_multitensor.gguf");
        std::fs::write(path, &file).unwrap();
        let m = unsafe { MmapModel::open(path) }.expect("open");
        assert_eq!(m.data_start(), data_start);
        assert_eq!(m.tensor_numel("norm"), Some(8));

        let nb = m.tensor_bytes("norm").unwrap();
        assert_eq!(nb.len(), 32);
        assert_eq!(f32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]), 1.0);

        let wb = m.tensor_bytes("w").unwrap();
        assert_eq!(wb.len(), 18);
        let dq = m.dequant("w").unwrap();
        assert_eq!(dq.len(), 32);
        assert!(dq.iter().all(|&v| v == -8.0));
        assert!(m.tensor_bytes("missing").is_none());

        drop(m);
        let _ = std::fs::remove_file(path);
    }
}
