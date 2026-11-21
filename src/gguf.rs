//! GGUF container format reader — **pure Rust, `std` only** (no llama.cpp, no deps).
//!
//! B1 slice `PH-101`/`PH-102`: header (magic · version · counts) → KV metadata of
//! every `GGUF_TYPE_*` (scalars, strings, nested arrays) → tensor-info table
//! (name · dims · ggml type · offset). mmap/mlock staged loading and dequant of the
//! tensor *data* are `PH-103`/`PH-104` (need `memmap2`, deferred off this slice so it
//! stays offline-buildable). Validation against a real model lands with `PH-106`.

use crate::core::DType;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF file magic ("GGUF" little-endian; versions 2 and 3).
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Default tensor-data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// True if the first four bytes are the GGUF magic.
pub fn matches_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == GGUF_MAGIC
}

/// `GGUF_TYPE_*` element kinds (scalar / string / array discriminator).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Array,
    U64,
    I64,
    F64,
}

impl ValueKind {
    fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::Str,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    fn code(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::Str => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }
}

/// A decoded KV value. `Array` keeps its element kind (may be a nested `Array`).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(ValueKind, Vec<Value>),
}

/// One row of the GGUF tensor-info table.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorEntry {
    pub name: String,
    pub dims: Vec<usize>,
    /// Raw `GGML_TYPE_*` code (lossless; many quant types are not in [`DType`] yet).
    pub ggml_type: u32,
    /// Best-effort mapped element type (`None` for unsupported/unknown codes).
    pub dtype: Option<DType>,
    /// Byte offset from the aligned tensor-data start (a multiple of `general.alignment`).
    pub offset: u64,
}

impl TensorEntry {
    /// Total element count = product of `dims` (1 for an empty dim list).
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Byte length of this tensor's data block (== `ggml_nbytes`), or `None` if its
    /// type/element-count isn't sized.
    pub fn nbytes(&self) -> Option<usize> {
        crate::quant::bytes_for_numel(self.ggml_type, self.numel())
    }
}

/// A parsed GGUF header: everything before the tensor data blob.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub n_tensors: u64,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorEntry>,
    /// Byte position where the (aligned) tensor data begins.
    pub data_offset: usize,
}

impl GgufFile {
    /// First metadata value for `key`, if present.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `general.alignment` (tensor-data byte alignment), defaulting to 32.
    pub fn alignment(&self) -> u64 {
        match self.get("general.alignment") {
            Some(Value::U32(a)) => u64::from(*a),
            Some(Value::U64(a)) => *a,
            _ => DEFAULT_ALIGNMENT,
        }
    }

    /// Metadata pair count (== `n_kv`).
    pub fn n_kv(&self) -> usize {
        self.metadata.len()
    }

    /// Aligned start of tensor data blob.
    pub fn aligned_data_start(&self) -> usize {
        let a = self.alignment().max(1) as usize;
        self.data_offset.div_ceil(a) * a
    }

    /// Byte range `[start, end)` for tensor data within the file.
    pub fn tensor_data_range(&self, name: &str) -> Option<(usize, usize)> {
        let t = self.tensors.iter().find(|t| t.name == name)?;
        let nbytes = t.nbytes()?;
        let start = self.aligned_data_start().checked_add(t.offset as usize)?;
        let end = start.checked_add(nbytes)?;
        Some((start, end))
    }
}

/// Header/parse failure with the byte offset where it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf parse error at {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Local result alias so this module's errors are not confused with `core::EngineError`.
pub type GgufResult<T> = std::result::Result<T, ParseError>;

/// Map a `GGML_TYPE_*` code to the engine [`DType`] subset we recognise.
pub fn ggml_dtype(code: u32) -> Option<DType> {
    Some(match code {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Q4_0,
        6 => DType::Q5_0,
        8 => DType::Q8_0,
        12 => DType::Q4_K,
        13 => DType::Q5_K,
        14 => DType::Q6_K,
        // Canonical ggml.h: I8 = 24, IQ1_M = 29, BF16 = 30. The MiniCPM-V 4.7 mmproj
        // (`PH-1201`) stores its vision weights as 30 — the earlier 24 → BF16 guess was
        // wrong (24 is I8) and is only reachable through BF16 now.
        30 => DType::BF16,
        _ => return None,
    })
}

/// Little-endian cursor over a GGUF byte buffer.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn need(&self, n: usize) -> GgufResult<()> {
        if self.pos + n > self.bytes.len() {
            return Err(ParseError {
                message: format!("unexpected EOF, need {n} bytes"),
                offset: self.pos,
            });
        }
        Ok(())
    }

    fn u8(&mut self) -> GgufResult<u8> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn take<const N: usize>(&mut self) -> GgufResult<[u8; N]> {
        self.need(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(a)
    }

    fn u16(&mut self) -> GgufResult<u16> {
        Ok(u16::from_le_bytes(self.take::<2>()?))
    }

    fn u32(&mut self) -> GgufResult<u32> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    fn u64(&mut self) -> GgufResult<u64> {
        Ok(u64::from_le_bytes(self.take::<8>()?))
    }

    fn f32(&mut self) -> GgufResult<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> GgufResult<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> GgufResult<String> {
        let len = self.u64()? as usize;
        self.need(len)?;
        let s =
            std::str::from_utf8(&self.bytes[self.pos..self.pos + len]).map_err(|e| ParseError {
                message: format!("invalid utf-8 string: {e}"),
                offset: self.pos,
            })?;
        self.pos += len;
        Ok(s.to_string())
    }

    /// Read a single value of the given kind (recurses for arrays).
    fn value(&mut self, kind: ValueKind) -> GgufResult<Value> {
        Ok(match kind {
            ValueKind::U8 => Value::U8(self.u8()?),
            ValueKind::I8 => Value::I8(self.u8()? as i8),
            ValueKind::U16 => Value::U16(self.u16()?),
            ValueKind::I16 => Value::I16(self.u16()? as i16),
            ValueKind::U32 => Value::U32(self.u32()?),
            ValueKind::I32 => Value::I32(self.u32()? as i32),
            ValueKind::F32 => Value::F32(self.f32()?),
            ValueKind::Bool => Value::Bool(self.u8()? != 0),
            ValueKind::Str => Value::Str(self.string()?),
            ValueKind::U64 => Value::U64(self.u64()?),
            ValueKind::I64 => Value::I64(self.u64()? as i64),
            ValueKind::F64 => Value::F64(self.f64()?),
            ValueKind::Array => {
                let elem_code = self.u32()?;
                let elem = ValueKind::from_code(elem_code).ok_or_else(|| ParseError {
                    message: format!("unknown array element kind {elem_code}"),
                    offset: self.pos,
                })?;
                let count = self.u64()?;
                // Malformed guard: every element is >= 1 byte, so a count larger
                // than the remaining buffer cannot be valid.
                if count > (self.bytes.len() - self.pos) as u64 {
                    return Err(ParseError {
                        message: format!("array count {count} exceeds remaining bytes"),
                        offset: self.pos,
                    });
                }
                let mut items = Vec::with_capacity(count.min(4096) as usize);
                for _ in 0..count {
                    items.push(self.value(elem)?);
                }
                Value::Array(elem, items)
            }
        })
    }
}

/// Parse a complete GGUF header (magic · version · KV metadata · tensor-info table).
///
/// Returns the byte offset where tensor data starts; the data itself is not read
/// here (that is `PH-103`/`PH-104`).
pub fn parse(bytes: &[u8]) -> GgufResult<GgufFile> {
    if !matches_magic(bytes) {
        return Err(ParseError {
            message: "missing GGUF magic".to_string(),
            offset: 0,
        });
    }
    let mut r = Reader::new(bytes);
    r.pos = 4;
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(ParseError {
            message: format!("unsupported GGUF version {version} (need 2 or 3)"),
            offset: 4,
        });
    }
    let (n_tensors, n_kv) = if version >= 3 {
        (r.u64()?, r.u64()?)
    } else {
        (u64::from(r.u32()?), u64::from(r.u32()?))
    };

    let mut metadata = Vec::with_capacity((n_kv).min(4096) as usize);
    for _ in 0..n_kv {
        let key = r.string()?;
        let code = r.u32()?;
        let kind = ValueKind::from_code(code).ok_or_else(|| ParseError {
            message: format!("unknown value kind {code} for key {key}"),
            offset: r.pos,
        })?;
        let value = r.value(kind)?;
        metadata.push((key, value));
    }

    let mut tensors = Vec::with_capacity((n_tensors).min(4096) as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd > 8 {
            return Err(ParseError {
                message: format!("tensor {name} declares {nd} dims (max 8)"),
                offset: r.pos,
            });
        }
        let mut dims = Vec::with_capacity(nd as usize);
        for _ in 0..nd {
            dims.push(r.u64()? as usize);
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(TensorEntry {
            name,
            dims,
            ggml_type,
            dtype: ggml_dtype(ggml_type),
            offset,
        });
    }

    Ok(GgufFile {
        version,
        n_tensors,
        metadata,
        tensors,
        data_offset: r.pos,
    })
}

/// Default `rusty inspect` read window — 16 MiB covers even very large tensor-name
/// tables, so the whole header parses without loading multi-GB tensor data.
pub const INSPECT_WINDOW: usize = 16 * 1024 * 1024;

/// Read a GGUF **header** from a file via a bounded, **std-only** prefix (no mmap).
///
/// Tensor-info tables are small; we read at most `window` bytes. If the header extends
/// past the window the parse fails on EOF — full mmap/mlock loading of tensor *data* is
/// `PH-103`. Powers `rusty inspect` (`PH-107`) and the file round-trip (`PH-106`).
pub fn parse_file(path: &Path, window: usize) -> io::Result<GgufFile> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let want = len.min(window);
    let mut buf = Vec::with_capacity(want);
    file.take(want as u64).read_to_end(&mut buf)?;
    parse(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read-only: extract raw tensor bytes from a GGUF file by header range.
pub fn read_tensor_bytes(path: &Path, header: &GgufFile, tensor_name: &str) -> io::Result<Vec<u8>> {
    let (start, end) = header.tensor_data_range(tensor_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("tensor {tensor_name} not found or unsized"),
        )
    })?;
    let mut file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if end > len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("tensor {tensor_name} range {start}..{end} beyond file size {len}"),
        ));
    }
    file.seek(SeekFrom::Start(start as u64))?;
    let mut buf = vec![0u8; end - start];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

// ───────────────────────── GGUF writer (GASV-C1) ─────────────────────────

/// Tensor payload for GGUF writing. Bytes must already be in the correct ggml layout.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    Raw(Vec<u8>),
    F32(Vec<f32>),
}

impl TensorData {
    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::Raw(b) => b.clone(),
            TensorData::F32(v) => {
                let mut out = Vec::with_capacity(v.len() * 4);
                for f in v {
                    out.extend_from_slice(&f.to_le_bytes());
                }
                out
            }
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            TensorData::Raw(b) => b.len(),
            TensorData::F32(v) => v.len() * 4,
        }
    }
}

/// Tensor to be written into a GGUF file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorWrite {
    pub name: String,
    pub dims: Vec<usize>,
    pub ggml_type: u32,
    pub data: TensorData,
}

/// Minimal std-only GGUF v3 writer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgufWriter {
    pub version: u32,
    pub alignment: u64,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorWrite>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteReport {
    pub bytes_written: usize,
    pub n_tensors: usize,
    pub n_kv: usize,
}

impl GgufWriter {
    pub fn new_v3() -> Self {
        Self {
            version: 3,
            alignment: DEFAULT_ALIGNMENT,
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment.max(1);
        self
    }

    pub fn push_kv(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.push((key.into(), value));
        self
    }

    pub fn push_tensor(mut self, tensor: TensorWrite) -> Self {
        self.tensors.push(tensor);
        self
    }

    pub fn add_kv(&mut self, key: impl Into<String>, value: Value) {
        self.metadata.push((key.into(), value));
    }

    pub fn add_tensor(&mut self, tensor: TensorWrite) {
        self.tensors.push(tensor);
    }

    fn effective_alignment(&self) -> u64 {
        match self.metadata.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, Value::U32(a))) => u64::from(*a).max(1),
            Some((_, Value::U64(a))) => (*a).max(1),
            _ => self.alignment.max(1),
        }
    }

    pub fn write_to_vec(&self) -> GgufResult<(Vec<u8>, WriteReport)> {
        if !(2..=3).contains(&self.version) {
            return Err(ParseError {
                message: format!("unsupported GGUF write version {}", self.version),
                offset: 0,
            });
        }
        let align = self.effective_alignment();
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        if self.version >= 3 {
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        } else {
            out.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
            out.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        }

        for (k, v) in &self.metadata {
            write_string(&mut out, k);
            write_value(&mut out, v)?;
        }

        // Precompute tensor data offsets relative to data start.
        let mut rel: u64 = 0;
        let mut tensor_bytes: Vec<Vec<u8>> = Vec::with_capacity(self.tensors.len());
        let mut offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let bytes = t.data.as_bytes();
            let expected = crate::quant::bytes_for_numel(t.ggml_type, t.dims.iter().product());
            if let Some(exp) = expected {
                if exp != bytes.len() {
                    return Err(ParseError {
                        message: format!(
                            "tensor {} byte size mismatch: expected {exp}, got {}",
                            t.name,
                            bytes.len()
                        ),
                        offset: out.len(),
                    });
                }
            }
            offsets.push(rel);
            rel = rel
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| ParseError {
                    message: "tensor offset overflow".to_string(),
                    offset: out.len(),
                })?;
            rel = align_up(rel, align);
            tensor_bytes.push(bytes);
        }

        for (t, off) in self.tensors.iter().zip(offsets.iter()) {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }

        let data_start = align_up(out.len() as u64, align) as usize;
        if data_start > out.len() {
            out.resize(data_start, 0);
        }

        for (i, bytes) in tensor_bytes.into_iter().enumerate() {
            let want = data_start as u64 + offsets[i];
            if want as usize > out.len() {
                out.resize(want as usize, 0);
            }
            out.extend_from_slice(&bytes);
            if i + 1 < self.tensors.len() {
                let next = data_start as u64 + offsets[i + 1];
                if next as usize > out.len() {
                    out.resize(next as usize, 0);
                }
            }
        }
        // final alignment padding optional; not required

        let report = WriteReport {
            bytes_written: out.len(),
            n_tensors: self.tensors.len(),
            n_kv: self.metadata.len(),
        };
        Ok((out, report))
    }

    pub fn write_to_file(&self, path: &Path) -> io::Result<WriteReport> {
        let (bytes, report) = self
            .write_to_vec()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, bytes)?;
        Ok(report)
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    if align <= 1 {
        return v;
    }
    v.div_ceil(align) * align
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &Value) -> GgufResult<()> {
    match v {
        Value::U8(x) => {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.push(*x);
        }
        Value::I8(x) => {
            out.extend_from_slice(&1u32.to_le_bytes());
            out.push(*x as u8);
        }
        Value::U16(x) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::I16(x) => {
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::I32(x) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::Bool(x) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*x));
        }
        Value::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, s);
        }
        Value::Array(elem_kind, items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&elem_kind.code().to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value_payload(out, item)?;
            }
        }
        Value::U64(x) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::I64(x) => {
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::F64(x) => {
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    Ok(())
}

fn write_value_payload(out: &mut Vec<u8>, v: &Value) -> GgufResult<()> {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::Str(s) => write_string(out, s),
        Value::Array(elem_kind, items) => {
            out.extend_from_slice(&elem_kind.code().to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value_payload(out, item)?;
            }
        }
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
    Ok(())
}

// ───────────────────── GASV-C2 skeleton (deterministic) ─────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkeletonTensorSpec {
    pub name: String,
    pub dims: Vec<usize>,
    pub ggml_type: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonSpec {
    pub arch: String,
    pub seed: u64,
    pub alignment: u64,
    pub context_length: u32,
    pub tensors: Vec<SkeletonTensorSpec>,
    pub kv: Vec<(String, Value)>,
}

#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        // Deterministic in [-1,1), avoid NaN/Inf.
        let bits = (self.next_u64() >> 40) as u32; // 24-bit mantissa
        let v = (bits as f32) / ((1u32 << 24) as f32); // [0,1)
        v * 2.0 - 1.0
    }
}

pub fn build_skeleton(spec: &SkeletonSpec) -> GgufResult<(Vec<u8>, WriteReport)> {
    let mut w = GgufWriter::new_v3().with_alignment(spec.alignment.max(1));
    w.add_kv("general.architecture", Value::Str(spec.arch.clone()));
    w.add_kv(
        "general.alignment",
        Value::U32(spec.alignment.max(1) as u32),
    );
    w.add_kv(
        format!("{}.context_length", spec.arch),
        Value::U32(spec.context_length),
    );
    w.add_kv("gasv.skeleton.seed", Value::U64(spec.seed));
    w.add_kv("gasv.skeleton.prng", Value::Str("splitmix64-f32-v1".into()));
    for (k, v) in &spec.kv {
        w.add_kv(k.clone(), v.clone());
    }
    let mut rng = SplitMix64::new(spec.seed);
    for t in &spec.tensors {
        let numel: usize = t.dims.iter().product();
        let data = match t.ggml_type {
            0 => {
                let mut v = Vec::with_capacity(numel);
                for _ in 0..numel {
                    v.push(rng.next_f32());
                }
                TensorData::F32(v)
            }
            _ => {
                let nbytes =
                    crate::quant::bytes_for_numel(t.ggml_type, numel).ok_or_else(|| {
                        ParseError {
                            message: format!(
                                "unsupported ggml_type {} for skeleton tensor {}",
                                t.ggml_type, t.name
                            ),
                            offset: 0,
                        }
                    })?;
                // Deterministic pseudo-random bytes for non-F32 skeletons (research/smoke only).
                let mut b = vec![0u8; nbytes];
                for byte in &mut b {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
                TensorData::Raw(b)
            }
        };
        w.add_tensor(TensorWrite {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            data,
        });
    }
    w.write_to_vec()
}

// ───────────────────── GASV-D3 export provenance ─────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct GasvExportProvenance {
    pub schema: String,
    pub artifact: String,
    pub source_gguf: Option<String>,
    pub source_manifest: Option<String>,
    pub arch: Option<String>,
    pub context_length: Option<u32>,
    pub quant: String,
    pub tensor_count: usize,
    pub seed: Option<u64>,
    pub gguf_sha256: Option<String>,
    pub bytes: usize,
}

impl GasvExportProvenance {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": self.schema,
            "artifact": self.artifact,
            "source_gguf": self.source_gguf,
            "source_manifest": self.source_manifest,
            "arch": self.arch,
            "context_length": self.context_length,
            "quant": self.quant,
            "tensor_count": self.tensor_count,
            "seed": self.seed,
            "gguf_sha256": self.gguf_sha256,
            "bytes": self.bytes,
        })
    }
}

/// Quantize F32 TensorWrite data to target ggml type (supported Q4_0/Q8_0).
/// Returns new TensorWrite; no silent coercion on unsupported types.
pub fn quantize_tensor_write(tw: &TensorWrite, target_ggml_type: u32) -> GgufResult<TensorWrite> {
    let f32s = match &tw.data {
        TensorData::F32(v) => v.clone(),
        TensorData::Raw(bytes) if tw.ggml_type == 0 => {
            let mut v = Vec::with_capacity(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            v
        }
        _ => {
            return Err(ParseError {
                message: format!(
                    "tensor {}: quantize export requires F32 source, got ggml_type {}",
                    tw.name, tw.ggml_type
                ),
                offset: 0,
            })
        }
    };
    let data = crate::quant::quantize(target_ggml_type, &f32s).map_err(|e| ParseError {
        message: format!("tensor {}: {}", tw.name, e.message),
        offset: 0,
    })?;
    let expected =
        crate::quant::bytes_for_numel(target_ggml_type, f32s.len()).ok_or_else(|| ParseError {
            message: format!(
                "tensor {}: cannot size ggml_type {target_ggml_type} for numel {}",
                tw.name,
                f32s.len()
            ),
            offset: 0,
        })?;
    if data.len() != expected {
        return Err(ParseError {
            message: format!(
                "tensor {}: quantized bytes {} != expected {}",
                tw.name,
                data.len(),
                expected
            ),
            offset: 0,
        });
    }
    Ok(TensorWrite {
        name: tw.name.clone(),
        dims: tw.dims.clone(),
        ggml_type: target_ggml_type,
        data: TensorData::Raw(data),
    })
}

// ───────────────────── GASV-D1 transplant (read-only source) ─────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransplantClass {
    DirectCopy,
    ConvertSafe,
    ProjectedResearch,
    Incompatible,
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransplantEntry {
    pub target_tensor: String,
    pub class: TransplantClass,
    pub source_tensor: String,
    pub expected_ggml_type: Option<u32>,
    pub expected_dims: Option<Vec<usize>>,
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransplantReport {
    pub target_tensor: String,
    pub class: TransplantClass,
    pub bytes: usize,
    pub sha256: String,
}

/// GASV-D2 policy gate. No silent coercion.
/// - DirectCopy/ConvertSafe: allowed by this gate (handler must still validate).
/// - ProjectedResearch: only with explicit owner-gated flag.
/// - Incompatible/Excluded: always rejected.
pub fn enforce_transplant_policy(
    entry: &TransplantEntry,
    allow_projected_research: bool,
) -> GgufResult<()> {
    match entry.class {
        TransplantClass::DirectCopy | TransplantClass::ConvertSafe => Ok(()),
        TransplantClass::ProjectedResearch if allow_projected_research => Ok(()),
        TransplantClass::ProjectedResearch => Err(ParseError {
            message: format!(
                "tensor {}: ProjectedResearch requires explicit owner gate",
                entry.target_tensor
            ),
            offset: 0,
        }),
        TransplantClass::Incompatible => Err(ParseError {
            message: format!(
                "tensor {}: Incompatible tensor rejected",
                entry.target_tensor
            ),
            offset: 0,
        }),
        TransplantClass::Excluded => Err(ParseError {
            message: format!("tensor {}: Excluded tensor rejected", entry.target_tensor),
            offset: 0,
        }),
    }
}

/// Validate and extract a direct-copy tensor from a source GGUF.
/// Rejects any class other than DirectCopy; verifies dims/type/sha256 when provided.
pub fn transplant_direct_copy(
    source_path: &Path,
    source_header: &GgufFile,
    entry: &TransplantEntry,
) -> GgufResult<(TensorWrite, TransplantReport)> {
    if entry.class != TransplantClass::DirectCopy {
        return Err(ParseError {
            message: format!(
                "tensor {}: class {:?} not allowed in direct_copy path",
                entry.target_tensor, entry.class
            ),
            offset: 0,
        });
    }
    let src = source_header
        .tensors
        .iter()
        .find(|t| t.name == entry.source_tensor)
        .ok_or_else(|| ParseError {
            message: format!(
                "tensor {}: source tensor {} not found",
                entry.target_tensor, entry.source_tensor
            ),
            offset: 0,
        })?;
    if let Some(t) = entry.expected_ggml_type {
        if src.ggml_type != t {
            return Err(ParseError {
                message: format!(
                    "tensor {}: ggml_type mismatch expected {t} got {}",
                    entry.target_tensor, src.ggml_type
                ),
                offset: 0,
            });
        }
    }
    if let Some(d) = &entry.expected_dims {
        if &src.dims != d {
            return Err(ParseError {
                message: format!(
                    "tensor {}: dims mismatch expected {:?} got {:?}",
                    entry.target_tensor, d, src.dims
                ),
                offset: 0,
            });
        }
    }
    let bytes =
        read_tensor_bytes(source_path, source_header, &entry.source_tensor).map_err(|e| {
            ParseError {
                message: format!("tensor {}: read failed: {e}", entry.target_tensor),
                offset: 0,
            }
        })?;
    let sha256 = crate::patch::sha256_hex(&bytes);
    if let Some(exp) = &entry.expected_sha256 {
        if !exp.eq_ignore_ascii_case(&sha256) {
            return Err(ParseError {
                message: format!(
                    "tensor {}: sha256 mismatch expected {exp} got {sha256}",
                    entry.target_tensor
                ),
                offset: 0,
            });
        }
    }
    let report = TransplantReport {
        target_tensor: entry.target_tensor.clone(),
        class: entry.class,
        bytes: bytes.len(),
        sha256: sha256.clone(),
    };
    let tw = TensorWrite {
        name: entry.target_tensor.clone(),
        dims: src.dims.clone(),
        ggml_type: src.ggml_type,
        data: TensorData::Raw(bytes),
    };
    Ok((tw, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gguf_header() {
        assert!(matches_magic(b"GGUF\x03\x00\x00\x00"));
    }

    #[test]
    fn rejects_other_headers() {
        assert!(!matches_magic(b"XXXX"));
        assert!(!matches_magic(b"GGU"));
    }

    fn push_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn push_u64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn push_str(v: &mut Vec<u8>, s: &str) {
        push_u64(v, s.len() as u64);
        v.extend_from_slice(s.as_bytes());
    }

    /// Build a minimal-but-real GGUF v3 header: 3 KV pairs + 1 tensor-info row.
    fn synth_gguf() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        push_u32(&mut b, 3); // version
        push_u64(&mut b, 1); // n_tensors
        push_u64(&mut b, 3); // n_kv
                             // general.architecture = "qwen2moe"
        push_str(&mut b, "general.architecture");
        push_u32(&mut b, 8); // Str
        push_str(&mut b, "qwen2moe");
        // general.block_count = 48 (u32)
        push_str(&mut b, "general.block_count");
        push_u32(&mut b, 4); // U32
        push_u32(&mut b, 48);
        // tokenizer.ggml.tokens = ["a","b"] (array of string)
        push_str(&mut b, "tokenizer.ggml.tokens");
        push_u32(&mut b, 9); // Array
        push_u32(&mut b, 8); // element kind = Str
        push_u64(&mut b, 2); // count
        push_str(&mut b, "a");
        push_str(&mut b, "b");
        // tensor info: blk.0.ffn_gate_exps.weight, dims [2048,512,64], Q4_K(14), offset 0
        push_str(&mut b, "blk.0.ffn_gate_exps.weight");
        push_u32(&mut b, 3); // n_dims
        push_u64(&mut b, 2048);
        push_u64(&mut b, 512);
        push_u64(&mut b, 64);
        push_u32(&mut b, 12); // ggml Q4_K
        push_u64(&mut b, 0); // offset
        b
    }

    #[test]
    fn parses_header_kv_and_tensor_info() {
        let bytes = synth_gguf();
        let f = parse(&bytes).expect("valid synthetic GGUF");
        assert_eq!(f.version, 3);
        assert_eq!(f.n_tensors, 1);
        assert_eq!(f.n_kv(), 3);
        assert_eq!(f.data_offset, bytes.len());

        assert!(matches!(f.get("general.architecture"), Some(Value::Str(s)) if s == "qwen2moe"));
        assert_eq!(f.get("general.block_count"), Some(&Value::U32(48)));

        match f.get("tokenizer.ggml.tokens") {
            Some(Value::Array(ValueKind::Str, items)) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], Value::Str(s) if s == "a"));
            }
            other => panic!("expected string array, got {other:?}"),
        }

        assert_eq!(f.tensors.len(), 1);
        let t = &f.tensors[0];
        assert_eq!(t.name, "blk.0.ffn_gate_exps.weight");
        assert_eq!(t.dims, vec![2048, 512, 64]);
        assert_eq!(t.ggml_type, 12);
        assert_eq!(t.dtype, Some(DType::Q4_K));
        assert_eq!(t.offset, 0);
    }

    #[test]
    fn ggml_dtype_maps_canonical_ggml_codes() {
        // Canonical GGML_TYPE_* enum: Q4_K=12, Q5_K=13, Q6_K=14 (a real Qwen2.5 file
        // proved the earlier 14/15/16 map was off by the K-types).
        assert_eq!(ggml_dtype(12), Some(DType::Q4_K));
        assert_eq!(ggml_dtype(13), Some(DType::Q5_K));
        assert_eq!(ggml_dtype(14), Some(DType::Q6_K));
        assert_eq!(ggml_dtype(0), Some(DType::F32));
        assert_eq!(ggml_dtype(1), Some(DType::F16));
        assert_eq!(ggml_dtype(8), Some(DType::Q8_0));
        assert_eq!(ggml_dtype(6), Some(DType::Q5_0)); // Q5_0 now mapped
                                                      // ggml.h tail: I8 = 24, IQ1_M = 29, BF16 = 30 — the MiniCPM-V 4.7 mmproj ships
                                                      // its vision weights as 30 (`PH-1201`); 24 was a stale BF16 guess.
        assert_eq!(ggml_dtype(30), Some(DType::BF16));
        assert_eq!(ggml_dtype(24), None); // I8 — not a float/quant DType
        assert_eq!(ggml_dtype(16), None); // IQ2_XXS, later band
    }

    #[test]
    fn tensor_entry_numel_and_nbytes_math() {
        // TensorEntry::nbytes == ggml_nbytes == numel/QK * block_bytes (bytes_for_numel).
        let f32 = TensorEntry {
            name: "n".into(),
            dims: vec![128],
            ggml_type: 0,
            dtype: Some(DType::F32),
            offset: 0,
        };
        assert_eq!(f32.numel(), 128);
        assert_eq!(f32.nbytes(), Some(512)); // 128*4
        let q40 = TensorEntry {
            name: "w".into(),
            dims: vec![256],
            ggml_type: 2,
            dtype: Some(DType::Q4_0),
            offset: 0,
        };
        assert_eq!(q40.nbytes(), Some(144)); // 256/32*18
        let q4k = TensorEntry {
            name: "e".into(),
            dims: vec![256],
            ggml_type: 12,
            dtype: Some(DType::Q4_K),
            offset: 0,
        };
        assert_eq!(q4k.nbytes(), Some(144)); // 256/256*144
        let mat = TensorEntry {
            name: "m".into(),
            dims: vec![896, 4864],
            ggml_type: 14,
            dtype: Some(DType::Q6_K),
            offset: 0,
        };
        assert_eq!(mat.numel(), 896 * 4864);
        assert_eq!(mat.nbytes(), Some((896 * 4864) / 256 * 210));
    }

    #[test]
    fn alignment_defaults_when_absent() {
        let f = parse(&synth_gguf()).unwrap();
        assert_eq!(f.alignment(), DEFAULT_ALIGNMENT);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = synth_gguf();
        b[0] = b'X';
        assert!(parse(&b).is_err());
    }

    #[test]
    fn rejects_truncated_buffer() {
        let full = synth_gguf();
        assert!(parse(&full[..full.len() / 2]).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut b = synth_gguf();
        // overwrite version field (bytes 4..8) with 1
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        let err = parse(&b).unwrap_err();
        assert!(err.message.contains("unsupported GGUF version"));
    }

    #[test]
    fn round_trips_a_real_file_via_std_io() {
        let bytes = synth_gguf();
        std::fs::create_dir_all("target").unwrap();
        let path = std::path::Path::new("target/rusty_gguf_fixture.gguf");
        std::fs::write(path, &bytes).unwrap();
        let f = parse_file(path, INSPECT_WINDOW).expect("parse_file reads header from disk");
        assert_eq!(f.version, 3);
        assert_eq!(f.n_tensors, 1);
        assert_eq!(f.n_kv(), 3);
        assert_eq!(f.data_offset, bytes.len());
        assert_eq!(f.tensors[0].name, "blk.0.ffn_gate_exps.weight");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_round_trip_header_and_f32_tensor() {
        let w = GgufWriter::new_v3()
            .push_kv("general.architecture", Value::Str("qwen35".into()))
            .push_kv("general.alignment", Value::U32(32))
            .push_kv("test.flag", Value::Bool(true))
            .push_tensor(TensorWrite {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![4],
                ggml_type: 0,
                data: TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            });
        let (bytes, report) = w.write_to_vec().expect("write");
        assert_eq!(report.n_tensors, 1);
        assert_eq!(report.n_kv, 3);
        let f = parse(&bytes).expect("parse written");
        assert_eq!(f.version, 3);
        assert_eq!(f.n_tensors, 1);
        assert_eq!(f.n_kv(), 3);
        assert_eq!(
            f.get("general.architecture"),
            Some(&Value::Str("qwen35".into()))
        );
        assert_eq!(f.tensors[0].name, "blk.0.attn_norm.weight");
        assert_eq!(f.tensors[0].dims, vec![4]);
        assert_eq!(f.tensors[0].ggml_type, 0);
        let data_start =
            ((f.data_offset as u64 + f.alignment() - 1) / f.alignment() * f.alignment()) as usize;
        let start = data_start + f.tensors[0].offset as usize;
        let nbytes = f.tensors[0].nbytes().unwrap();
        let mut got = Vec::with_capacity(4);
        for chunk in bytes[start..start + nbytes].chunks_exact(4) {
            got.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn writer_round_trip_raw_bytes_and_alignment() {
        let raw = vec![0u8; 144]; // one Q4_K block worth of bytes
        let w = GgufWriter::new_v3()
            .with_alignment(64)
            .push_kv("general.alignment", Value::U32(64))
            .push_kv("general.architecture", Value::Str("test".into()))
            .push_tensor(TensorWrite {
                name: "t0".into(),
                dims: vec![256],
                ggml_type: 12,
                data: TensorData::Raw(raw.clone()),
            })
            .push_tensor(TensorWrite {
                name: "t1".into(),
                dims: vec![2],
                ggml_type: 0,
                data: TensorData::F32(vec![0.5, -0.5]),
            });
        let (bytes, _) = w.write_to_vec().expect("write");
        let f = parse(&bytes).expect("parse");
        assert_eq!(f.alignment(), 64);
        assert_eq!(f.tensors.len(), 2);
        let data_start = ((f.data_offset as u64).div_ceil(64) * 64) as usize;
        assert_eq!(data_start % 64, 0);
        assert_eq!(f.tensors[0].offset % 64, 0);
        assert_eq!(f.tensors[1].offset % 64, 0);
        let s0 = data_start + f.tensors[0].offset as usize;
        assert_eq!(&bytes[s0..s0 + raw.len()], raw.as_slice());
    }

    #[test]
    fn skeleton_is_deterministic_and_parseable() {
        let spec = SkeletonSpec {
            arch: "qwen35".into(),
            seed: 42,
            alignment: 32,
            context_length: 4096,
            tensors: vec![
                SkeletonTensorSpec {
                    name: "blk.0.attn_norm.weight".into(),
                    dims: vec![8],
                    ggml_type: 0,
                },
                SkeletonTensorSpec {
                    name: "blk.0.ffn_gate.weight".into(),
                    dims: vec![4, 8],
                    ggml_type: 0,
                },
            ],
            kv: vec![("general.name".into(), Value::Str("gasv-test".into()))],
        };
        let (b1, r1) = build_skeleton(&spec).expect("build1");
        let (b2, _r2) = build_skeleton(&spec).expect("build2");
        assert_eq!(b1, b2, "same seed/spec must be deterministic");
        assert_eq!(r1.n_tensors, 2);
        let f = parse(&b1).expect("parse");
        assert_eq!(f.get("gasv.skeleton.seed"), Some(&Value::U64(42)));
        assert_eq!(
            f.get("gasv.skeleton.prng"),
            Some(&Value::Str("splitmix64-f32-v1".into()))
        );
        assert_eq!(f.tensors.len(), 2);
    }

    #[test]
    fn skeleton_smoke_validate_parse_manifest_load_dequant() {
        let spec = SkeletonSpec {
            arch: "gasvtest".into(),
            seed: 7,
            alignment: 32,
            context_length: 128,
            tensors: vec![SkeletonTensorSpec {
                name: "tok.weight".into(),
                dims: vec![4, 8],
                ggml_type: 0,
            }],
            kv: vec![],
        };
        let (bytes, report) = build_skeleton(&spec).expect("build");
        assert_eq!(report.n_tensors, 1);
        // parse header
        let f = parse(&bytes).expect("parse");
        assert_eq!(
            f.get("general.architecture"),
            Some(&Value::Str("gasvtest".into()))
        );
        // write/read file + inspect-style parse_file
        std::fs::create_dir_all("target").unwrap();
        let path = std::path::Path::new("target/gasv_skeleton_smoke.gguf");
        std::fs::write(path, &bytes).unwrap();
        let f2 = parse_file(path, INSPECT_WINDOW).expect("parse_file");
        assert_eq!(f2.tensors[0].name, "tok.weight");
        // smoke load/dequant via MmapModel
        let model = unsafe { crate::loader::MmapModel::open(path) }.expect("open");
        let vals = model.dequant("tok.weight").expect("dequant");
        assert_eq!(vals.len(), 32);
        assert!(vals.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn transplant_direct_copy_round_trip_and_rejects_bad_class() {
        let src_w = GgufWriter::new_v3()
            .push_kv("general.architecture", Value::Str("test".into()))
            .push_kv("general.alignment", Value::U32(32))
            .push_tensor(TensorWrite {
                name: "a.weight".into(),
                dims: vec![4],
                ggml_type: 0,
                data: TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            });
        let (src_bytes, _) = src_w.write_to_vec().unwrap();
        std::fs::create_dir_all("target").unwrap();
        let src_path = std::path::Path::new("target/gasv_transplant_src.gguf");
        std::fs::write(src_path, &src_bytes).unwrap();
        let src_header = parse(&src_bytes).unwrap();
        let entry = TransplantEntry {
            target_tensor: "b.weight".into(),
            class: TransplantClass::DirectCopy,
            source_tensor: "a.weight".into(),
            expected_ggml_type: Some(0),
            expected_dims: Some(vec![4]),
            expected_sha256: None,
        };
        let (tw, report) = transplant_direct_copy(src_path, &src_header, &entry).unwrap();
        assert_eq!(report.class, TransplantClass::DirectCopy);
        assert_eq!(report.bytes, 16);
        let out_w = GgufWriter::new_v3()
            .push_kv("general.architecture", Value::Str("test".into()))
            .push_kv("general.alignment", Value::U32(32))
            .push_tensor(tw);
        let (out_bytes, _) = out_w.write_to_vec().unwrap();
        let out_h = parse(&out_bytes).unwrap();
        let ds = ((out_h.data_offset as u64).div_ceil(32) * 32) as usize;
        let s = ds + out_h.tensors[0].offset as usize;
        let mut got = Vec::new();
        for chunk in out_bytes[s..s + 16].chunks_exact(4) {
            got.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
        let bad = TransplantEntry {
            class: TransplantClass::Incompatible,
            ..entry
        };
        assert!(transplant_direct_copy(src_path, &src_header, &bad).is_err());
        let _ = std::fs::remove_file(src_path);
    }

    #[test]
    fn transplant_policy_rejects_incompatible_and_gates_research() {
        let base = TransplantEntry {
            target_tensor: "t".into(),
            class: TransplantClass::DirectCopy,
            source_tensor: "s".into(),
            expected_ggml_type: None,
            expected_dims: None,
            expected_sha256: None,
        };
        assert!(enforce_transplant_policy(&base, false).is_ok());
        let conv = TransplantEntry {
            class: TransplantClass::ConvertSafe,
            ..base.clone()
        };
        assert!(enforce_transplant_policy(&conv, false).is_ok());
        let research = TransplantEntry {
            class: TransplantClass::ProjectedResearch,
            ..base.clone()
        };
        assert!(enforce_transplant_policy(&research, false).is_err());
        assert!(enforce_transplant_policy(&research, true).is_ok());
        let bad = TransplantEntry {
            class: TransplantClass::Incompatible,
            ..base.clone()
        };
        assert!(enforce_transplant_policy(&bad, true).is_err());
        let ex = TransplantEntry {
            class: TransplantClass::Excluded,
            ..base
        };
        assert!(enforce_transplant_policy(&ex, true).is_err());
    }

    #[test]
    fn gasv_d3_quantize_export_q4_0_and_provenance() {
        let src = TensorWrite {
            name: "w".into(),
            dims: vec![32],
            ggml_type: 0,
            data: TensorData::F32((0..32).map(|i| (i as f32 - 16.0) / 8.0).collect()),
        };
        let q = quantize_tensor_write(&src, 2).expect("q4_0");
        assert_eq!(q.ggml_type, 2);
        assert_eq!(q.data.byte_len(), 18);
        let w = GgufWriter::new_v3()
            .push_kv("general.architecture", Value::Str("gasvtest".into()))
            .push_kv("gasv.skeleton.seed", Value::U64(7))
            .push_kv("gasv.export.quant", Value::Str("Q4_0".into()))
            .push_tensor(q);
        let (bytes, report) = w.write_to_vec().expect("write");
        let prov = GasvExportProvenance {
            schema: "gasv.export-provenance.v1".into(),
            artifact: "gsv-versus-ai.gguf".into(),
            source_gguf: Some("source.gguf".into()),
            source_manifest: Some("gasv.transplant-manifest.v1".into()),
            arch: Some("gasvtest".into()),
            context_length: Some(4096),
            quant: "Q4_0".into(),
            tensor_count: report.n_tensors,
            seed: Some(7),
            gguf_sha256: Some(crate::patch::sha256_hex(&bytes)),
            bytes: report.bytes_written,
        };
        let j = prov.to_json();
        assert_eq!(j["schema"], "gasv.export-provenance.v1");
        assert_eq!(j["quant"], "Q4_0");
        assert_eq!(j["tensor_count"], 1);
        assert!(j["gguf_sha256"].as_str().unwrap().len() == 64);
        // round-trip parse
        let f = parse(&bytes).expect("parse");
        assert_eq!(f.tensors[0].ggml_type, 2);
    }
}
