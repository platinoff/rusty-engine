//! Core value types: data types, tensor descriptors, unified error/result.

use std::fmt;

/// Quantized + float element types the engine understands (GGUF dtype subset).
/// GGUF block names (`Q4_K` …) are kept verbatim, hence the camel-case allow.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q5_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    I8,
}

impl DType {
    /// Bytes per element for float types; block formats report `None` (handled
    /// per GGUF block in B1).
    pub fn element_bytes(self) -> Option<usize> {
        match self {
            DType::F32 => Some(4),
            DType::F16 | DType::BF16 => Some(2),
            DType::I8
            | DType::Q4_0
            | DType::Q5_0
            | DType::Q4_K
            | DType::Q5_K
            | DType::Q6_K
            | DType::Q8_0 => None,
        }
    }
}

/// A named tensor read from the model container.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Unified engine error (B0 placeholder; enriched with quant/IO variants in B1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError {
    pub message: String,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rusty-engine error: {}", self.message)
    }
}

impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_is_product_of_dims() {
        let t = TensorInfo {
            name: "blk.0.attn_q".into(),
            dims: vec![4096, 1024],
            dtype: DType::Q4_K,
        };
        assert_eq!(t.numel(), 4096 * 1024);
    }

    #[test]
    fn float_types_report_element_bytes() {
        assert_eq!(DType::F32.element_bytes(), Some(4));
        assert_eq!(DType::Q4_K.element_bytes(), None);
    }
}
