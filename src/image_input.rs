//! PH-1205: std-only image decoders for the multimodal chat wire.
//!
//! MVP formats: binary PPM (P6) and uncompressed 24/32-bpp BMP. No `image`
//! crate, no Python. The decoded RGB is normalized to f32 and packed into
//! `d_model`-wide rows for the existing vision-prefix path. These are raw
//! pixel vectors, **not** trained embeddings — no image→answer claim (G13).

/// Row-major 8-bit RGB image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

fn le_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn le_i32(b: &[u8], at: usize) -> i32 {
    le_u32(b, at) as i32
}

/// Skip PPM whitespace and `#` comment lines, return the next token.
fn ppm_token(bytes: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    loop {
        while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < bytes.len() && bytes[*pos] == b'#' {
            while *pos < bytes.len() && bytes[*pos] != b'\n' {
                *pos += 1;
            }
            continue;
        }
        break;
    }
    let start = *pos;
    while *pos < bytes.len() && !bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
    if start == *pos {
        None
    } else {
        Some(bytes[start..*pos].to_vec())
    }
}

fn ppm_usize(bytes: &[u8], pos: &mut usize, name: &str) -> Result<usize, String> {
    let tok = ppm_token(bytes, pos).ok_or_else(|| format!("ppm: missing {name}"))?;
    let s = std::str::from_utf8(&tok).map_err(|_| format!("ppm: bad {name}"))?;
    s.parse::<usize>().map_err(|_| format!("ppm: bad {name}"))
}

/// Decode a binary P6 PPM (maxval 255) into row-major RGB.
pub fn decode_ppm(bytes: &[u8]) -> Result<RgbImage, String> {
    let mut pos = 0usize;
    let magic = ppm_token(bytes, &mut pos).ok_or("ppm: truncated header")?;
    if magic != b"P6" {
        return Err("ppm: not P6".into());
    }
    let width = ppm_usize(bytes, &mut pos, "width")?;
    let height = ppm_usize(bytes, &mut pos, "height")?;
    let maxval = ppm_usize(bytes, &mut pos, "maxval")?;
    if width == 0 || height == 0 {
        return Err("ppm: zero dimension".into());
    }
    if maxval != 255 {
        return Err("ppm: only maxval 255 is supported".into());
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_whitespace() {
        return Err("ppm: missing raster separator".into());
    }
    pos += 1; // exactly one whitespace byte precedes the raster
    let need = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(3))
        .ok_or("ppm: dimensions overflow")?;
    let end = pos.checked_add(need).ok_or("ppm: raster overflow")?;
    if end > bytes.len() {
        return Err("ppm: truncated raster".into());
    }
    Ok(RgbImage {
        width,
        height,
        rgb: bytes[pos..end].to_vec(),
    })
}

/// Decode an uncompressed BITMAPINFOHEADER BMP (24/32 bpp, bottom-up or
/// top-down) into row-major RGB. Alpha on 32 bpp is dropped.
pub fn decode_bmp(bytes: &[u8]) -> Result<RgbImage, String> {
    if bytes.len() < 54 || &bytes[..2] != b"BM" {
        return Err("bmp: bad signature".into());
    }
    let off = le_u32(bytes, 10) as usize;
    if le_u32(bytes, 14) != 40 {
        return Err("bmp: only BITMAPINFOHEADER is supported".into());
    }
    let w = le_i32(bytes, 18);
    let h = le_i32(bytes, 22);
    let bpp = le_u16(bytes, 28) as usize;
    if w <= 0 || h == 0 {
        return Err("bmp: bad dimensions".into());
    }
    if le_u32(bytes, 30) != 0 || (bpp != 24 && bpp != 32) {
        return Err("bmp: unsupported compression or bpp".into());
    }
    let width = w as usize;
    let height = h.unsigned_abs() as usize;
    let top_down = h > 0;
    let stride = (width * bpp).div_ceil(32) * 4;
    let end = off
        .checked_add(stride.checked_mul(height).ok_or("bmp: stride overflow")?)
        .ok_or("bmp: raster overflow")?;
    if end > bytes.len() {
        return Err("bmp: truncated raster".into());
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for row_out in 0..height {
        let row_in = if top_down {
            row_out
        } else {
            height - 1 - row_out
        };
        let base = off + row_in * stride;
        for x in 0..width {
            let p = base + x * (bpp / 8);
            rgb.extend_from_slice(&[bytes[p + 2], bytes[p + 1], bytes[p]]);
        }
    }
    Ok(RgbImage { width, height, rgb })
}

/// Dispatch by magic bytes: `P6` → PPM, `BM` → BMP.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage, String> {
    if bytes.starts_with(b"P6") {
        decode_ppm(bytes)
    } else if bytes.starts_with(b"BM") {
        decode_bmp(bytes)
    } else {
        Err("image: unsupported format (need P6 PPM or BMP)".into())
    }
}

/// Normalized row-major RGB f32, packed little-endian into `d_model`-wide
/// rows (zero-padded at the end). Structural wire bytes only — no trained
/// vision→text alignment is implied.
pub fn image_prefix_bytes(images: &[RgbImage], d_model: usize) -> Result<Vec<u8>, String> {
    if d_model == 0 {
        return Err("image_prefix: zero d_model".into());
    }
    let mut values: Vec<f32> = Vec::new();
    for img in images {
        if img.width == 0 || img.height == 0 {
            return Err("image_prefix: empty image".into());
        }
        values.extend(img.rgb.iter().map(|v| *v as f32 / 255.0));
    }
    if values.is_empty() {
        return Err("image_prefix: no pixels".into());
    }
    while values.len() % d_model != 0 {
        values.push(0.0);
    }
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}
