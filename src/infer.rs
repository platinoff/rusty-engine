//! Inference plumbing: generation options, math primitives, attention + sampler, and a
//! paged KV-cache.
//!
//! Band B3 primitives that need **no model** live here as pure-`f32`, unit-tested code:
//! `PH-301` RMSNorm, `PH-302` attention (RoPE + causal scaled-dot + GQA broadcast), `PH-303`
//! paged KV-cache, `PH-304` sampler chain. End-to-end dense generation (`PH-305`) wires these
//! against a real GGUF and lands with `PH-306`.

use crate::moe::softmax;
use std::cmp::Ordering;

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

/// Dot product of two equal-length vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Root-mean-square layer norm: `x * rsqrt(mean(x^2)+eps) * weight`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len(), "rmsnorm: shape mismatch");
    let n = x.len() as f32;
    let ms = if n == 0.0 {
        0.0
    } else {
        x.iter().map(|v| v * v).sum::<f32>() / n
    };
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Rotary positional embedding (RoPE) over one head vector of even length, at `pos`.
///
/// Pair `i` (elements `2i, 2i+1`) rotates by `pos * theta_i`,
/// `theta_i = base^(-2i/d)` — the llama.cpp / HF convention.
pub fn rope(head: &[f32], pos: usize, base: f32) -> Vec<f32> {
    let d = head.len();
    let mut out = head.to_vec();
    for i in 0..d / 2 {
        let theta = 1.0 / base.powf((2 * i) as f32 / d.max(1) as f32);
        let angle = pos as f32 * theta;
        let (c, s) = (angle.cos(), angle.sin());
        let (x0, x1) = (head[2 * i], head[2 * i + 1]);
        out[2 * i] = x0 * c - x1 * s;
        out[2 * i + 1] = x1 * c + x0 * s;
    }
    out
}

/// Apply a causal mask (top-right triangle → `-inf`) to an in-place `seq×seq` score grid.
pub fn apply_causal_mask(scores: &mut [f32], seq: usize) {
    assert_eq!(scores.len(), seq * seq, "causal mask: not seq×seq");
    for (i, row) in scores.chunks_mut(seq).enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            if j > i {
                *x = f32::NEG_INFINITY;
            }
        }
    }
}

/// Single-head causal scaled-dot-product attention.
///
/// `q`,`k`,`v` are row-major `seq×d` buffers. Row `i` attends to `j<=i` only
/// (`softmax(q·k/√d)` over the causal span, then weighted sum of `v`).
pub fn attention_causal(q: &[f32], k: &[f32], v: &[f32], seq: usize, d: usize) -> Vec<f32> {
    if seq == 0 || d == 0 {
        return Vec::new();
    }
    assert_eq!(q.len(), seq * d, "attention: q shape");
    assert_eq!(k.len(), seq * d, "attention: k shape");
    assert_eq!(v.len(), seq * d, "attention: v shape");
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; seq * d];
    for (i, qrow) in q.chunks(d).enumerate() {
        let mut row: Vec<f32> = Vec::with_capacity(seq);
        for (j, krow) in k.chunks(d).enumerate() {
            let s = dot(qrow, krow) * scale;
            row.push(if j > i { f32::NEG_INFINITY } else { s });
        }
        let probs = softmax(&row);
        let orow = &mut out[i * d..(i + 1) * d];
        for (j, &p) in probs.iter().enumerate() {
            let vrow = &v[j * d..(j + 1) * d];
            for (o, b) in orow.iter_mut().zip(vrow) {
                *o += p * *b;
            }
        }
    }
    out
}

/// Grouped-query expansion: `kv` (`seq × n_kv × dims`) → `seq × n_q × dims` by repeating
/// each KV head `n_q / n_kv` times (MQA when `n_kv == 1`, MHA when `n_kv == n_q`).
pub fn gqa_repeat_kv(kv: &[f32], seq: usize, n_kv: usize, n_q: usize, dims: usize) -> Vec<f32> {
    assert_eq!(kv.len(), seq * n_kv * dims, "gqa: kv shape");
    assert!(dims >= 1, "gqa: dims must be > 0");
    assert!(
        n_kv >= 1 && n_q % n_kv == 0,
        "gqa: n_q must be a multiple of n_kv"
    );
    let f = n_q / n_kv;
    let mut out = Vec::with_capacity(seq * n_q * dims);
    for tok in kv.chunks(n_kv * dims) {
        for head in tok.chunks(dims) {
            for _ in 0..f {
                out.extend_from_slice(head);
            }
        }
    }
    out
}

/// Index of the maximum logit (`None` for an empty slice).
pub fn argmax(logits: &[f32]) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
}

/// Softmax temperature scaling (`temp<=0` is a no-op; greedy handled by [`argmax`]).
pub fn apply_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    if temp <= 0.0 {
        return logits.to_vec();
    }
    logits.iter().map(|&v| v / temp).collect()
}

/// Zero out all but the `k` largest logits (the rest → `-inf`).
pub fn mask_top_k(logits: &[f32], k: usize) -> Vec<f32> {
    if k == 0 || logits.is_empty() {
        return logits.to_vec();
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let threshold = sorted[(k - 1).min(sorted.len() - 1)];
    logits
        .iter()
        .map(|&v| if v >= threshold { v } else { f32::NEG_INFINITY })
        .collect()
}

/// Nucleus (top-p) masking: keep the smallest set of highest-probability logits whose
/// cumulative softmax mass reaches `p`; the rest → `-inf`. `p>=1` is a no-op.
pub fn mask_top_p(logits: &[f32], p: f32) -> Vec<f32> {
    if p >= 1.0 || logits.is_empty() {
        return logits.to_vec();
    }
    let probs = softmax(logits);
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));
    let mut keep = vec![f32::NEG_INFINITY; logits.len()];
    let mut cum = 0.0f32;
    for &i in &order {
        keep[i] = logits[i];
        cum += probs[i];
        if cum >= p {
            break;
        }
    }
    keep
}

/// Repetition penalty on already-generated tokens (llama.cpp-style): divide positive logits
/// and multiply negative ones by `penalty` (no-op when `penalty == 1`).
pub fn apply_repetition_penalty(logits: &[f32], tokens: &[usize], penalty: f32) -> Vec<f32> {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return logits.to_vec();
    }
    let mut out = logits.to_vec();
    for &t in tokens {
        if let Some(v) = out.get_mut(t) {
            *v = if *v > 0.0 { *v / penalty } else { *v * penalty };
        }
    }
    out
}

/// Draw a token id from `logits` under `opts` given a deterministic uniform draw `r` in
/// `[0,1)`. Greedy when `temperature<=0`. Production feeds an RNG for `r`.
pub fn sample(logits: &[f32], opts: &GenerationOptions, r: f32) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    if opts.temperature <= 0.0 {
        return argmax(logits);
    }
    let scaled = apply_temperature(logits, opts.temperature);
    let kmasked = mask_top_k(&scaled, opts.top_k);
    let pmasked = mask_top_p(&kmasked, opts.top_p);
    let probs = softmax(&pmasked);
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let mut cum = 0.0f32;
        for (i, &pr) in probs.iter().enumerate() {
            cum += pr / sum;
            if cum >= r {
                return Some(i);
            }
        }
    }
    argmax(logits)
}

/// Paged KV-cache: token KV rows are stored in fixed-size pages (vLLM idea) so long
/// contexts avoid one huge contiguous allocation. Each page holds `page_size` tokens of
/// `[k(dims) | v(dims)]`; `append`/`get` address by logical token index.
#[derive(Debug)]
pub struct PagedKvCache {
    page_size: usize,
    dims: usize,
    pages: Vec<Vec<f32>>,
    len: usize,
}

impl PagedKvCache {
    /// New cache with `page_size` tokens per page and `dims` = per-token head dimension.
    pub fn new(page_size: usize, dims: usize) -> Self {
        assert!(
            page_size > 0 && dims > 0,
            "PagedKvCache: page_size/dims must be > 0"
        );
        Self {
            page_size,
            dims,
            pages: Vec::new(),
            len: 0,
        }
    }

    /// Append a token's key and value; returns its index. Rolls to a new page when full.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> usize {
        assert_eq!(k.len(), self.dims, "append: k width");
        assert_eq!(v.len(), self.dims, "append: v width");
        if self.len % self.page_size == 0 {
            self.pages
                .push(vec![0.0f32; self.page_size * 2 * self.dims]);
        }
        let page = &mut self.pages[self.len / self.page_size];
        let off = (self.len % self.page_size) * 2 * self.dims;
        page[off..off + self.dims].copy_from_slice(k);
        page[off + self.dims..off + 2 * self.dims].copy_from_slice(v);
        self.len += 1;
        self.len - 1
    }

    /// Number of cached tokens.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read back a token's `(key, value)`, or `None` if out of range.
    pub fn get(&self, idx: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        if idx >= self.len {
            return None;
        }
        let page = self.pages.get(idx / self.page_size)?;
        let off = (idx % self.page_size) * 2 * self.dims;
        let k = page[off..off + self.dims].to_vec();
        let v = page[off + self.dims..off + 2 * self.dims].to_vec();
        Some((k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn defaults_are_conservative() {
        let o = GenerationOptions::default();
        assert_eq!(o.max_tokens, 256);
        assert!((o.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn rmsnorm_normalizes() {
        let out = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        assert!((out[0] - 3.0 / 12.5f32.sqrt()).abs() < EPS);
        assert!((out[1] - 4.0 / 12.5f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn dot_and_argmax_temperature() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < EPS);
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), Some(1));
        assert_eq!(
            apply_temperature(&[2.0, 4.0, 6.0], 2.0),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn rope_identity_at_pos_zero() {
        let head = [1.0, 0.0, 0.5, -0.5];
        assert_eq!(rope(&head, 0, 10000.0), head.to_vec());
    }

    #[test]
    fn rope_rotates_first_pair_by_one_radian() {
        // i=0 => theta=1, pos=1 => angle=1rad: [1,0] -> [cos1, sin1]
        let out = rope(&[1.0, 0.0], 1, 10000.0);
        assert!((out[0] - 1.0f32.cos()).abs() < EPS);
        assert!((out[1] - 1.0f32.sin()).abs() < EPS);
    }

    #[test]
    fn causal_mask_triangles() {
        let mut s = [9.0f32; 4];
        apply_causal_mask(&mut s, 2);
        assert_eq!(s, [9.0, f32::NEG_INFINITY, 9.0, 9.0]);
    }

    #[test]
    fn attention_causal_first_row_is_own_value() {
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 3.0, 4.0, 5.0];
        let out = attention_causal(&q, &k, &v, 2, 2);
        // row 0 attends only to itself -> exactly v0
        assert!((out[0] - 2.0).abs() < EPS);
        assert!((out[1] - 3.0).abs() < EPS);
        // row 1 mixes v0,v1, stays between them
        assert!(out[2] > 2.0 && out[2] < 4.0);
        assert!(out[3] > 3.0 && out[3] < 5.0);
    }

    #[test]
    fn gqa_broadcasts_kv_heads() {
        // 2 tokens, 1 kv-head, 2 q-heads, dims=2 -> each head repeated twice
        let out = gqa_repeat_kv(&[1.0, 2.0, 3.0, 4.0], 2, 1, 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn paged_kv_roundtrip_across_pages() {
        let mut c = PagedKvCache::new(2, 2);
        assert!(c.is_empty());
        c.append(&[1.0, 1.0], &[1.0, 2.0]); // token 0
        c.append(&[2.0, 2.0], &[2.0, 4.0]); // token 1 (page 0 full)
        c.append(&[3.0, 3.0], &[3.0, 6.0]); // token 2 (page 1)
        assert_eq!(c.len(), 3);
        assert_eq!(c.get(0), Some((vec![1.0, 1.0], vec![1.0, 2.0])));
        assert_eq!(c.get(2), Some((vec![3.0, 3.0], vec![3.0, 6.0])));
        assert_eq!(c.get(3), None);
    }

    #[test]
    fn sampler_basics() {
        let m = mask_top_k(&[1.0, 2.0, 3.0, 4.0], 2);
        assert!(m[0].is_infinite() && m[1].is_infinite() && m[2] > 0.0 && m[3] > 0.0);
        let out = apply_repetition_penalty(&[2.0, -2.0, 2.0], &[0, 1], 2.0);
        assert!((out[0] - 1.0).abs() < EPS);
        assert!((out[1] + 4.0).abs() < EPS);
        let opts = GenerationOptions {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&[1.0, 5.0, 3.0], &opts, 0.5), Some(1));
        let warm = GenerationOptions {
            temperature: 1.0,
            top_k: 40,
            top_p: 1.0,
            max_tokens: 8,
        };
        assert_eq!(sample(&[0.0, 0.0, 10.0], &warm, 0.5), Some(2));
    }
}
