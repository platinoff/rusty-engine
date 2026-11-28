//! PH-1205: multimodal HTTP wire — std-only PPM/BMP decode + `messages[].images`
//! base64 extraction. Structural/no-model gate: pixel vectors are consumed
//! verbatim, with no trained alignment and no image→answer claim (G13 blocked).

use rusty_engine::core::Result;
use rusty_engine::image_input::{decode_bmp, decode_image, decode_ppm, image_prefix_bytes};
use rusty_engine::serve::extract_images_b64;
use rusty_engine::textgen::{chat_response, chat_response_with_images, Completes};

fn ppm_2x2() -> Vec<u8> {
    let mut b = b"P6\n2 2\n255\n".to_vec();
    b.extend_from_slice(&[
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        255, 255, 255, // white
    ]);
    b
}

/// Minimal top-down 24-bpp BMP: 14-byte file header + 40-byte DIB + two 8-byte rows.
fn bmp_2x2() -> Vec<u8> {
    let mut b = vec![0u8; 54 + 16];
    b[0..2].copy_from_slice(b"BM");
    b[10..14].copy_from_slice(&54u32.to_le_bytes());
    b[14..18].copy_from_slice(&40u32.to_le_bytes());
    b[18..22].copy_from_slice(&2i32.to_le_bytes());
    b[22..26].copy_from_slice(&2i32.to_le_bytes()); // positive height = top-down
    b[26..28].copy_from_slice(&1u16.to_le_bytes());
    b[28..30].copy_from_slice(&24u16.to_le_bytes());
    let pixels: [[u8; 3]; 4] = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 255]];
    // Rows are 8-byte aligned (width 2 x 24bpp -> 6 bytes + 2 padding).
    for row in 0..2 {
        for x in 0..2 {
            let at = 54 + row * 8 + x * 3;
            b[at..at + 3].copy_from_slice(&pixels[row * 2 + x]);
        }
    }
    b
}

const EXPECTED_RGB: [u8; 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];

#[test]
fn ppm_p6_decodes_row_major_rgb() {
    let img = decode_ppm(&ppm_2x2()).unwrap();
    assert_eq!((img.width, img.height), (2, 2));
    assert_eq!(img.rgb, EXPECTED_RGB);
}

#[test]
fn bmp_24bpp_top_down_decodes_bgr_to_rgb() {
    let img = decode_bmp(&bmp_2x2()).unwrap();
    assert_eq!((img.width, img.height), (2, 2));
    assert_eq!(img.rgb, EXPECTED_RGB);
}

#[test]
fn image_dispatch_rejects_unknown_magic() {
    assert!(decode_image(b"hello").is_err());
    assert!(decode_image(&ppm_2x2()).is_ok());
    assert!(decode_image(&bmp_2x2()).is_ok());
}

#[test]
fn pixel_rows_are_normalized_and_padded_to_hidden_width() {
    let img = decode_ppm(&ppm_2x2()).unwrap();
    let bytes = image_prefix_bytes(&[img], 3).unwrap();
    assert_eq!(bytes.len(), 4 * 3 * 4); // 4 rows x dim 3 x f32
    let first = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(first, 1.0);
    // Row 1 starts at float index 3; its green component is float index 4.
    let green = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert_eq!(green, 1.0);
}

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | chunk.get(2).copied().unwrap_or(0) as u32;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[test]
fn images_base64_are_extracted_from_messages_without_touching_other_fields() {
    let body = format!(
        r#"{{"model":"x","messages":[{{"role":"user","content":"what","images":["{}"]}}]}}"#,
        b64(&ppm_2x2())
    );
    let images = extract_images_b64(&body).unwrap();
    assert_eq!(images.len(), 1);
    assert_eq!(images[0], ppm_2x2());
    assert!(extract_images_b64(r#"{"messages":[]}"#).unwrap().is_empty());
    assert!(extract_images_b64(r#"{"messages":[{"images":["bad!"]}]}"#).is_err());
}

struct VisionStub;

impl Completes for VisionStub {
    fn complete(&self, _json: &str, _max_new: usize) -> Result<String> {
        Ok("text".into())
    }
    fn token_len(&self, text: &str) -> usize {
        text.len()
    }
    fn complete_with_images(
        &self,
        _json: &str,
        _max_new: usize,
        images: &[Vec<u8>],
    ) -> Result<String> {
        Ok(format!("images={}", images.len()))
    }
}

#[test]
fn chat_response_dispatches_images_to_the_engine() {
    let body = format!(
        r#"{{"messages":[{{"role":"user","content":"x","images":["{}"]}}]}}"#,
        b64(&ppm_2x2())
    );
    let out = chat_response_with_images(&VisionStub, &body, 4, &[ppm_2x2()]);
    assert_eq!(out.response.status, 200);
    assert!(out.response.body.contains("images=1"));

    // Without the images wire the default path stays text-only.
    let plain = chat_response(&VisionStub, &body, 4);
    assert_eq!(plain.response.status, 200);
    assert!(plain.response.body.contains("text"));
}
