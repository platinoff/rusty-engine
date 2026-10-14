//! GGUF container format reader — **pure Rust, `std` only** (no llama.cpp, no deps).
//!
//! B1 slice `PH-101`/`PH-102`: header (magic · version · counts) → KV metadata of
//! every `GGUF_TYPE_*` (scalars, strings, nested arrays) → tensor-info table
//! (name · dims · ggml type · offset). mmap/mlock staged loading and dequant of the
//! tensor *data* are `PH-103`/`PH-104` (need `memmap2`, deferred off this slice so it
//! stays offline-buildable). Validation against a real model lands with `PH-106`.

use crate::core::DType;
use std::fs::File;
use std::io::{self, Read};
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
    /// Offset, in tensor-data units, from the aligned data start.
    pub offset: u64,
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
        8 => DType::Q8_0,
        14 => DType::Q4_K,
        15 => DType::Q5_K,
        16 => DType::Q6_K,
        24 => DType::BF16,
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
        push_u32(&mut b, 14); // ggml Q4_K
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
        assert_eq!(t.ggml_type, 14);
        assert_eq!(t.dtype, Some(DType::Q4_K));
        assert_eq!(t.offset, 0);
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
}
