//! What each vertex is painted: its material and its base colour (the
//! material's factor times its base-colour texture at the vertex's UV).
//!
//! A generated character is one fused surface, so a belt, a buckle or a
//! pouch is not a part of its own; it shows only in the paint. The
//! checker reads the paint to find such parts (`owner.rs`) and to draw
//! close-ups a person recognises. PNG textures are decoded; any other
//! codec gives the material's factor alone.

use crate::scene::Model;
use serde_json::Value;
use std::collections::BTreeMap;

pub struct Paint {
    /// Material index per welded vertex (`usize::MAX`: none).
    pub material: Vec<usize>,
    /// Linear-ish RGB in [0, 1] per welded vertex.
    pub rgb: Vec<[f32; 3]>,
    /// Whether any texture was read (else colours are factors only).
    pub textured: bool,
}

struct Image {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

fn decode_png(bytes: &[u8]) -> Option<Image> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().ok()?;
    let mut buf = vec![0u8; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        _ => return None,
    };
    let mut rgb = Vec::with_capacity(w * h * 3);
    for px in buf[..w * h * ch].chunks(ch) {
        match ch {
            1 | 2 => rgb.extend([px[0], px[0], px[0]]),
            _ => rgb.extend([px[0], px[1], px[2]]),
        }
    }
    Some(Image { w, h, rgb })
}

fn image_bytes<'a>(model: &'a Model, img: &Value) -> Option<&'a [u8]> {
    let bv = img.get("bufferView")?.as_u64()? as usize;
    let v = model.glb.arr("bufferViews").get(bv)?;
    let off = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
    let len = v.get("byteLength")?.as_u64()? as usize;
    model.glb.bin.get(off..off + len)
}

pub fn read(model: &Model) -> Paint {
    let nv = model.nverts();
    let mut sum = vec![[0f64; 3]; nv];
    let mut cnt = vec![0usize; nv];
    let mut material = vec![usize::MAX; nv];
    let mut images: BTreeMap<usize, Option<Image>> = BTreeMap::new();
    let mut textured = false;
    let mats = model.glb.arr("materials");
    for prim in &model.prims {
        let p = &model.glb.json["meshes"][prim.mesh]["primitives"][prim.prim];
        let mi = p.get("material").and_then(Value::as_u64).map(|x| x as usize);
        let m = mi.and_then(|i| mats.get(i));
        let pbr = m.and_then(|m| m.get("pbrMetallicRoughness"));
        let factor: [f64; 3] = match pbr.and_then(|p| p.get("baseColorFactor")).and_then(Value::as_array) {
            Some(a) if a.len() >= 3 => [a[0].as_f64().unwrap_or(1.0), a[1].as_f64().unwrap_or(1.0), a[2].as_f64().unwrap_or(1.0)],
            _ => [1.0; 3],
        };
        let tex = pbr.and_then(|p| p.get("baseColorTexture"));
        let tex_img = tex
            .and_then(|t| t.get("index"))
            .and_then(Value::as_u64)
            .and_then(|ti| model.glb.arr("textures").get(ti as usize))
            .and_then(|t| t.get("source"))
            .and_then(Value::as_u64)
            .map(|x| x as usize);
        let set = tex.and_then(|t| t.get("texCoord")).and_then(Value::as_u64).unwrap_or(0);
        let uv: Option<Vec<f64>> = p
            .get("attributes")
            .and_then(|a| a.get(format!("TEXCOORD_{set}")))
            .and_then(Value::as_u64)
            .and_then(|a| model.glb.read_f64(a as usize).ok())
            .filter(|(n, _)| *n == 2)
            .map(|x| x.1);
        if let Some(ii) = tex_img {
            images.entry(ii).or_insert_with(|| {
                let img = model.glb.arr("images").get(ii)?;
                let mime = img.get("mimeType").and_then(Value::as_str).unwrap_or("");
                let bytes = image_bytes(model, img)?;
                (mime == "image/png" || bytes.starts_with(b"\x89PNG")).then(|| decode_png(bytes)).flatten()
            });
        }
        let image = tex_img.and_then(|ii| images.get(&ii)).and_then(|x| x.as_ref());
        textured |= image.is_some() && uv.is_some();
        for (ri, &wi) in prim.raw_to_weld.iter().enumerate() {
            let v = wi as usize;
            let mut c = factor;
            if let (Some(img), Some(uv)) = (image, &uv) {
                if let (Some(&u), Some(&t)) = (uv.get(ri * 2), uv.get(ri * 2 + 1)) {
                    let x = ((u.rem_euclid(1.0)) * img.w as f64) as usize;
                    let y = ((t.rem_euclid(1.0)) * img.h as f64) as usize;
                    let o = (y.min(img.h - 1) * img.w + x.min(img.w - 1)) * 3;
                    for k in 0..3 {
                        c[k] *= img.rgb[o + k] as f64 / 255.0;
                    }
                }
            }
            for k in 0..3 {
                sum[v][k] += c[k];
            }
            cnt[v] += 1;
            if material[v] == usize::MAX {
                material[v] = mi.unwrap_or(usize::MAX - 1);
            }
        }
    }
    let rgb = sum
        .iter()
        .zip(&cnt)
        .map(|(s, &n)| if n > 0 { [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32] } else { [0.6; 3] })
        .collect();
    Paint { material, rgb, textured }
}

/// Patches of one material and near-one colour: each vertex joins a
/// neighbour's patch when it shares the material and sits within `tol`
/// (RGB distance) of that patch's mean so far. Returns patch per vertex.
pub fn patches(model: &Model, paint: &Paint, tol: f32) -> Vec<u32> {
    let nv = model.nverts();
    let mut id = vec![u32::MAX; nv];
    let mut next = 0u32;
    for s in 0..nv {
        if id[s] != u32::MAX {
            continue;
        }
        id[s] = next;
        let mut mean = [paint.rgb[s][0] as f64, paint.rgb[s][1] as f64, paint.rgb[s][2] as f64];
        let mut n = 1.0;
        let mut stack = vec![s];
        while let Some(v) = stack.pop() {
            for &u in &model.adj[v] {
                let u = u as usize;
                if id[u] != u32::MAX || paint.material[u] != paint.material[s] {
                    continue;
                }
                let c = paint.rgb[u];
                let d = ((c[0] as f64 - mean[0]).powi(2) + (c[1] as f64 - mean[1]).powi(2) + (c[2] as f64 - mean[2]).powi(2)).sqrt();
                if d <= tol as f64 {
                    id[u] = next;
                    n += 1.0;
                    for k in 0..3 {
                        mean[k] += (c[k] as f64 - mean[k]) / n;
                    }
                    stack.push(u);
                }
            }
        }
        next += 1;
    }
    id
}
