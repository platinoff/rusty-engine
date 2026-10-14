//! Inference plumbing: generation options, paged KV-cache, sampler.
//!
//! B0 defines the option shape only. Attention/RoPE (PH-302), the paged KV-cache
//! (PH-303) and the sampler chain (PH-304) are filled in band B3.

/// Sampling / decode options (OpenAI-compat surface).
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 40,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_conservative() {
        let o = GenerationOptions::default();
        assert_eq!(o.max_tokens, 256);
        assert!((o.temperature - 1.0).abs() < f32::EPSILON);
    }
}
