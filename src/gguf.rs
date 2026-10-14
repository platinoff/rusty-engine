//! GGUF container detection. The full v3 header + tensor-info parser lands in
//! band B1 (PH-101/102); this scaffold only recognises the magic so downstream
//! code can dispatch on container format.

/// GGUF file magic ("GGUF" little-endian, versions 2/3).
pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// True if the first four bytes are the GGUF magic.
pub fn matches_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == GGUF_MAGIC
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
}
