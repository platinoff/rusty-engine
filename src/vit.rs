//! PH-1202: pure-Rust ViT tower for MiniCPM-V, built on the existing
//! `QMat`/`matvec` kernels. Structural/unit gate only: deterministic toy
//! weights and shape/residual checks, **no** real GGUF tensors, no golden
//! `llama.cpp-omni` mtmd parity yet, and no image→answer or quality claim
//! (G13 is blocked). Golden fixpoints remain a follow-up band.

use crate::qmat::QMat;

/// Tiny-to-real ViT geometry: `grid × grid` patches of `patch × patch × 3`,
/// each embedded into `dim`, `heads` attention heads, GELU MLP `dim → mlp → dim`.
#[derive(Debug, Clone, Copy)]
pub struct VitGeom {
    pub patch: usize,
    pub grid: usize,
    pub dim: usize,
    pub heads: usize,
    pub mlp: usize,
    pub layers: usize,
    pub eps: f32,
}

impl VitGeom {
    pub fn n_tokens(&self) -> usize {
        self.grid * self.grid
    }
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

/// LayerNorm: `(x - mean) / sqrt(var + eps) * w + b` (not RMSNorm — ViT
/// blocks carry a bias, unlike the LLM RMSNorm blocks).
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "layer_norm: weight width");
    assert_eq!(x.len(), b.len(), "layer_norm: bias width");
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(w.iter().zip(b))
        .map(|(v, (wv, bv))| (v - mean) * inv * wv + bv)
        .collect()
}

/// GELU tanh approximation, the GGML-style kernel used for vision MLPs.
pub fn gelu(x: f32) -> f32 {
    const K: f32 = 0.797_884_6; // sqrt(2/PI)
    0.5 * x * (1.0 + (K * (x + 0.044_715 * x * x * x)).tanh())
}

/// Embed row-major `H × W × 3` pixels into `grid²` row-major patch vectors.
/// Each patch is flattened row-major (`patch × patch × 3`) and projected by `w`.
pub fn patch_embed(image: &[f32], geom: &VitGeom, w: &QMat, bias: &[f32]) -> Vec<f32> {
    let h = geom.grid * geom.patch;
    let patch_vals = geom.patch * geom.patch * 3;
    assert_eq!(bias.len(), geom.dim, "patch_embed: bias width");
    assert_eq!(image.len(), h * h * 3, "patch_embed: image size");
    assert_eq!(w.rows(), geom.dim, "patch_embed: projection rows");
    assert_eq!(w.cols(), patch_vals, "patch_embed: projection columns");

    let mut out = Vec::with_capacity(geom.n_tokens() * geom.dim);
    for py in 0..geom.grid {
        for px in 0..geom.grid {
            let mut flat = Vec::with_capacity(patch_vals);
            for dy in 0..geom.patch {
                for dx in 0..geom.patch {
                    let pixel = (py * geom.patch + dy) * h + px * geom.patch + dx;
                    let off = pixel * 3;
                    flat.extend_from_slice(&image[off..off + 3]);
                }
            }
            let projected = w.matvec(&flat);
            out.extend(projected.iter().zip(bias).map(|(v, b)| *v + *b));
        }
    }
    out
}

/// Bilinear interpolation of a `table_grid² × dim` learned position table to
/// `target_grid² × dim` (the 2D position interpolation used when the input
/// grid differs from the trained grid).
pub fn pos_embed_2d(table: &[f32], table_grid: usize, target_grid: usize, dim: usize) -> Vec<f32> {
    assert!(table_grid > 0 && target_grid > 0 && dim > 0);
    assert_eq!(table.len(), table_grid * table_grid * dim);
    let coord = |t: usize| -> (usize, usize, f32) {
        if target_grid == 1 {
            return (0, 0, 0.0);
        }
        let f = (t * (table_grid - 1)) as f32 / (target_grid - 1) as f32;
        let lo = f.floor() as usize;
        (lo, (lo + 1).min(table_grid - 1), f - lo as f32)
    };
    let mut out = Vec::with_capacity(target_grid * target_grid * dim);
    for ty in 0..target_grid {
        let (y0, y1, wy) = coord(ty);
        for tx in 0..target_grid {
            let (x0, x1, wx) = coord(tx);
            for d in 0..dim {
                let v00 = table[(y0 * table_grid + x0) * dim + d];
                let v01 = table[(y0 * table_grid + x1) * dim + d];
                let v10 = table[(y1 * table_grid + x0) * dim + d];
                let v11 = table[(y1 * table_grid + x1) * dim + d];
                out.push(
                    (1.0 - wx) * (1.0 - wy) * v00
                        + wx * (1.0 - wy) * v01
                        + (1.0 - wx) * wy * v10
                        + wx * wy * v11,
                );
            }
        }
    }
    out
}

/// One ViT block's weights (toy fixtures own them; real GGUF loading is a
/// separate band). `q/k/v/o` are `dim × dim`, MLP is `dim → mlp → dim`.
#[derive(Debug, Clone)]
pub struct VitBlockWeights {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub q: QMat,
    pub k: QMat,
    pub v: QMat,
    pub o: QMat,
    /// Per-output-dimension biases from the real `attn_*` tensors (empty for
    /// synthetic fixtures that want an unbiased projection).
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub o_bias: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub mlp_in: QMat,
    pub mlp_out: QMat,
}

/// Per-token LayerNorm: `w`/`b` are `dim`-wide and applied to each token row.
fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % w.len(), 0, "layer_norm_rows: weight width");
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(w.len()) {
        out.extend(layer_norm(row, w, b, eps));
    }
    out
}

/// Average-pool a `grid × grid` token map into `(grid/factor)²` tokens,
/// preserving channel order. `factor=2` is the 4× compressor, `factor=4`
/// is the 16× compressor used by MiniCPM-V's visual merger.
pub fn visual_pool(tokens: &[f32], grid: usize, dim: usize, factor: usize) -> Vec<f32> {
    assert!(
        factor > 0 && grid > 0 && dim > 0,
        "visual_pool: bad geometry"
    );
    assert!(
        grid % factor == 0,
        "visual_pool: grid not divisible by factor"
    );
    assert_eq!(tokens.len(), grid * grid * dim, "visual_pool: token width");
    let out_grid = grid / factor;
    let mut out = Vec::with_capacity(out_grid * out_grid * dim);
    for by in 0..out_grid {
        for bx in 0..out_grid {
            for d in 0..dim {
                let mut sum = 0.0f32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        let y = by * factor + dy;
                        let x = bx * factor + dx;
                        sum += tokens[(y * grid + x) * dim + d];
                    }
                }
                out.push(sum / (factor * factor) as f32);
            }
        }
    }
    out
}

/// Linear projection into the LLM hidden size, followed by a per-token
/// LayerNorm (`ds_ln` in the merger). Both matrices stay on `QMat::matvec`.
#[derive(Debug, Clone)]
pub struct VisualProjector {
    pub project: QMat,
    pub bias: Vec<f32>,
    pub ln_w: Vec<f32>,
    pub ln_b: Vec<f32>,
}

/// Pooled tokens `[n × dim]` → `[n × hidden]`, normalized per token.
pub fn visual_project(tokens: &[f32], p: &VisualProjector, eps: f32) -> Vec<f32> {
    assert_eq!(p.bias.len(), p.project.rows(), "visual_project: bias width");
    assert_eq!(p.ln_w.len(), p.project.rows(), "visual_project: norm width");
    assert_eq!(p.ln_b.len(), p.project.rows(), "visual_project: norm bias");
    let projected = matvec_rows(&p.project, tokens);
    let with_bias: Vec<f32> = projected
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + p.bias[i % p.bias.len()])
        .collect();
    layer_norm_rows(&with_bias, &p.ln_w, &p.ln_b, eps)
}

/// Full visual merger: spatial pooling → hidden projection → LayerNorm.
#[derive(Debug, Clone)]
pub struct VisualCompressor {
    pub grid: usize,
    pub dim: usize,
    /// `2` = 4× token reduction, `4` = 16×.
    pub factor: usize,
    pub projector: VisualProjector,
}

pub fn visual_merger_forward(tokens: &[f32], c: &VisualCompressor, eps: f32) -> Vec<f32> {
    let pooled = visual_pool(tokens, c.grid, c.dim, c.factor);
    visual_project(&pooled, &c.projector, eps)
}

/// Add a `cols`-wide bias to each token row (empty bias = no-op).
fn add_bias_rows(x: &[f32], bias: &[f32]) -> Vec<f32> {
    if bias.is_empty() {
        return x.to_vec();
    }
    assert_eq!(x.len() % bias.len(), 0, "add_bias_rows: bias width");
    x.iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % bias.len()])
        .collect()
}

/// Apply a `rows × cols` matrix to each `cols`-wide token row.
fn matvec_rows(w: &QMat, tokens: &[f32]) -> Vec<f32> {
    let cols = w.cols();
    assert_eq!(tokens.len() % cols, 0, "matvec_rows: token width");
    let mut out = Vec::with_capacity(tokens.len() / cols * w.rows());
    for row in tokens.chunks_exact(cols) {
        out.extend(w.matvec(row));
    }
    out
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in row.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = row.iter().sum();
    for v in row.iter_mut() {
        *v /= sum;
    }
}

/// Non-causal multi-head self-attention over `grid²` tokens. Input rows are
/// already LayerNorm'ed with the position embedding added. Returns flattened
/// `tokens × dim`, with the `o` projection applied per token.
pub fn vit_attention_forward(tokens: &[f32], w: &VitBlockWeights, geom: &VitGeom) -> Vec<f32> {
    let dim = geom.dim;
    let hd = geom.head_dim();
    assert_eq!(tokens.len() % dim, 0, "vit_attention: token buffer width");
    let n = tokens.len() / dim;
    assert!(
        hd > 0 && hd * geom.heads == dim,
        "vit_attention: head geometry"
    );

    let mut q = Vec::with_capacity(n * dim);
    let mut k = Vec::with_capacity(n * dim);
    let mut v = Vec::with_capacity(n * dim);
    for row in tokens.chunks_exact(dim) {
        q.extend(w.q.matvec(row));
        k.extend(w.k.matvec(row));
        v.extend(w.v.matvec(row));
    }
    let q = add_bias_rows(&q, &w.q_bias);
    let k = add_bias_rows(&k, &w.k_bias);
    let v = add_bias_rows(&v, &w.v_bias);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut ctx = vec![0.0f32; n * dim];
    for h in 0..geom.heads {
        let hs = h * hd;
        for i in 0..n {
            let qi = &q[i * dim + hs..i * dim + hs + hd];
            let mut scores = vec![0.0f32; n];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * dim + hs..j * dim + hs + hd];
                *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax(&mut scores);
            for (j, p) in scores.iter().enumerate() {
                let vj = &v[j * dim + hs..j * dim + hs + hd];
                for (o, value) in ctx[i * dim + hs..i * dim + hs + hd].iter_mut().zip(vj) {
                    *o += *p * *value;
                }
            }
        }
    }

    let mut out = Vec::with_capacity(n * dim);
    for row in ctx.chunks_exact(dim) {
        out.extend(w.o.matvec(row));
    }
    add_bias_rows(&out, &w.o_bias)
}

/// `ln1 → attention → residual → ln2 → GELU MLP → residual`.
pub fn vit_block_forward(x: &[f32], w: &VitBlockWeights, geom: &VitGeom) -> Vec<f32> {
    let n1 = layer_norm_rows(x, &w.ln1_w, &w.ln1_b, geom.eps);
    let attn = vit_attention_forward(&n1, w, geom);
    let h: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let n2 = layer_norm_rows(&h, &w.ln2_w, &w.ln2_b, geom.eps);
    let hidden: Vec<f32> = matvec_rows(&w.mlp_in, &n2).into_iter().map(gelu).collect();
    let ffn = matvec_rows(&w.mlp_out, &hidden);
    h.iter().zip(&ffn).map(|(a, b)| a + b).collect()
}

/// Full tower: `n` block steps followed by the final LayerNorm (`post_ln`).
pub fn vit_tower_forward(
    tokens: &[f32],
    blocks: &[VitBlockWeights],
    post_w: &[f32],
    post_b: &[f32],
    geom: &VitGeom,
) -> Vec<f32> {
    assert_eq!(
        tokens.len(),
        geom.n_tokens() * geom.dim,
        "vit tower: token width"
    );
    assert_eq!(blocks.len(), geom.layers, "vit tower: layer count");
    let mut x = tokens.to_vec();
    for block in blocks {
        x = vit_block_forward(&x, block, geom);
    }
    layer_norm_rows(&x, post_w, post_b, geom.eps)
}
