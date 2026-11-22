//! Minimal safetensors reader for the NanoJev checkpoint bundle:
//! 8-byte little-endian header length, JSON header (`{name: {dtype, shape, data_offsets}}`),
//! raw tensor bytes. Tensors are F32 only in this checkpoint.
//!
//! The file is mmap'd; tensor views borrow from the mapping (zero-copy).

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::core::{EngineError, Result};

#[derive(Debug, Clone)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    /// Absolute byte range into the mmap (already includes the header offset).
    beg: usize,
    end: usize,
}

/// An mmap'd `.safetensors` file. Parses the JSON header eagerly; tensor
/// payloads stay zero-copy in the mapping.
pub struct SafeTensors {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Open and parse a safetensors file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| EngineError {
            message: format!("safetensors: cannot open {}: {e}", path.display()),
        })?;
        // SAFETY: the file is opened read-only and never written to during this process.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| EngineError {
            message: format!("safetensors: cannot mmap {}: {e}", path.display()),
        })?;
        if mmap.len() < 8 {
            return Err(EngineError {
                message: "safetensors: file smaller than header prefix".into(),
            });
        }
        let mut len_buf = [0u8; 8];
        len_buf.copy_from_slice(&mmap[..8]);
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let header_start = 8usize;
        // safetensors spec: header JSON is padded so tensor data starts on an
        // 8-byte boundary. The prefix stores the unpadded JSON length.
        let padded_header_len = header_len.div_ceil(8) * 8;
        let data_base = header_start
            .checked_add(padded_header_len)
            .ok_or(EngineError {
                message: "safetensors: header length overflows".into(),
            })?;
        let header_end = header_start.checked_add(header_len).ok_or(EngineError {
            message: "safetensors: header length overflows".into(),
        })?;
        if header_end > mmap.len() {
            return Err(EngineError {
                message: "safetensors: header exceeds file length".into(),
            });
        }
        let header_bytes = &mmap[header_start..header_end];
        let header: serde_json::Value =
            serde_json::from_slice(header_bytes).map_err(|e| EngineError {
                message: format!("safetensors: header JSON: {e}"),
            })?;
        let obj = header.as_object().ok_or(EngineError {
            message: "safetensors: header is not a JSON object".into(),
        })?;
        let mut tensors = HashMap::with_capacity(obj.len());
        for (name, tv) in obj {
            let dtype = tv
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let shape: Vec<usize> = tv
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                })
                .ok_or(EngineError {
                    message: format!("safetensors: tensor {name}: missing shape"),
                })?;
            let offsets = tv
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or(EngineError {
                    message: format!("safetensors: tensor {name}: missing data_offsets"),
                })?;
            if offsets.len() != 2 {
                return Err(EngineError {
                    message: format!("safetensors: tensor {name}: data_offsets must be [beg, end]"),
                });
            }
            let beg_rel = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
            let end_rel = offsets[1].as_u64().unwrap_or(u64::MAX) as usize;
            let beg = data_base.checked_add(beg_rel).ok_or(EngineError {
                message: format!("safetensors: tensor {name}: offset overflow"),
            })?;
            let end = data_base.checked_add(end_rel).ok_or(EngineError {
                message: format!("safetensors: tensor {name}: offset overflow"),
            })?;
            if end > mmap.len() || beg >= end {
                return Err(EngineError {
                    message: format!("safetensors: tensor {name}: byte range out of bounds"),
                });
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    beg,
                    end,
                },
            );
        }
        Ok(Self { mmap, tensors })
    }

    /// Number of tensors in the file.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Sorted tensor names (for diagnostics).
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| EngineError {
            message: format!("safetensors: tensor '{name}' not found"),
        })
    }

    /// Shape of a tensor.
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.info(name)?.shape)
    }

    /// Zero-copy F32 view of a tensor.
    ///
    /// # Errors
    /// Returns an error if the tensor is missing, not F32, or not 4-byte aligned.
    pub fn f32_view(&self, name: &str) -> Result<(&[usize], &[f32])> {
        let info = self.info(name)?;
        if info.dtype != "F32" {
            return Err(EngineError {
                message: format!(
                    "safetensors: tensor '{name}' dtype {} (expected F32)",
                    info.dtype
                ),
            });
        }
        let ptr = self.mmap.as_ptr() as usize + info.beg;
        if ptr % std::mem::align_of::<f32>() != 0 {
            return Err(EngineError {
                message: format!("safetensors: tensor '{name}' not 4-byte aligned"),
            });
        }
        let count: usize = info.shape.iter().product();
        let byte_len = count * std::mem::size_of::<f32>();
        if info.beg + byte_len > info.end || info.end > self.mmap.len() {
            return Err(EngineError {
                message: format!("safetensors: tensor '{name}' shape exceeds byte range"),
            });
        }
        // SAFETY: pointer is 4-byte aligned (checked) and within the mmap; the file
        // is mapped read-only and outlives the returned slice's borrow of `self`.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, count) };
        Ok((&info.shape, slice))
    }

    /// Copy one F32 tensor into an owned Vec (fallback for hosts where the
    /// zero-copy view cannot be used, e.g. alignment edge cases).
    pub fn f32_owned(&self, name: &str) -> Result<Vec<f32>> {
        let (_, slice) = self.f32_view(name)?;
        Ok(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn header_roundtrip_offsets() {
        // Synthetic: 8-byte len + JSON header + 4×f32 payload, 16 bytes total data.
        let header = r#"{"x":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header.as_bytes());
        // safetensors pads the header JSON to an 8-byte boundary before data.
        while (8 + buf.len()) % 8 != 0 {
            buf.push(b' ');
        }
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        let dir = std::env::temp_dir().join("safetensors_unit");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        std::fs::write(&path, &buf).unwrap();
        let st = crate::nanojev::safetensors::SafeTensors::open(&path).unwrap();
        assert_eq!(st.tensor_count(), 1);
        let (shape, view) = st.f32_view("x").unwrap();
        assert_eq!(shape, &[2, 2]);
        assert_eq!(view, &[1.0, 2.0, 3.0, 4.0]);
    }
}
