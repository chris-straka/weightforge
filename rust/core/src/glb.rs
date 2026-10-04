//! GLB container + accessor IO on raw JSON, so unknown fields, extensions,
//! materials, and animations survive a round trip untouched. Writing only
//! ever replaces skinning attribute data (weightforge never moves vertices).

use serde_json::{Value, json};
use std::fmt;

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

pub fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

pub struct Glb {
    pub json: Value,
    pub bin: Vec<u8>,
}

const MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Glb {
    pub fn read_path(path: &std::path::Path) -> Result<Glb> {
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Glb::parse(&bytes)
    }

    pub fn parse(b: &[u8]) -> Result<Glb> {
        if b.len() < 20 || u32_at(b, 0) != MAGIC {
            return err("not a glTF binary (.glb) file");
        }
        if u32_at(b, 4) != 2 {
            return err("glTF container version is not 2");
        }
        let total = (u32_at(b, 8) as usize).min(b.len());
        let mut off = 12;
        let mut json = None;
        let mut bin = Vec::new();
        while off + 8 <= total {
            let len = u32_at(b, off) as usize;
            let kind = u32_at(b, off + 4);
            let start = off + 8;
            if start + len > total {
                return err("GLB chunk runs past end of file");
            }
            let data = &b[start..start + len];
            match kind {
                CHUNK_JSON => json = Some(serde_json::from_slice::<Value>(data).map_err(|e| Error(format!("GLB JSON chunk: {e}")))?),
                CHUNK_BIN if bin.is_empty() => bin = data.to_vec(),
                _ => {}
            }
            off = start + ((len + 3) & !3);
        }
        let json = json.ok_or_else(|| Error("GLB has no JSON chunk".into()))?;
        Ok(Glb { json, bin })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut js = serde_json::to_vec(&self.json).expect("json");
        while js.len() % 4 != 0 {
            js.push(b' ');
        }
        let mut bin = self.bin.clone();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let has_bin = !bin.is_empty();
        let total = 12 + 8 + js.len() + if has_bin { 8 + bin.len() } else { 0 };
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(js.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&js);
        if has_bin {
            out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
            out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
            out.extend_from_slice(&bin);
        }
        out
    }

    pub fn arr(&self, key: &str) -> &[Value] {
        self.json.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
    }

    fn accessor_layout(&self, idx: usize) -> Result<Layout> {
        let acc = self.arr("accessors").get(idx).ok_or_else(|| Error(format!("accessor {idx} missing")))?;
        if acc.get("sparse").is_some() {
            return err(format!("accessor {idx} is sparse (unsupported)"));
        }
        let count = acc.get("count").and_then(Value::as_u64).unwrap_or(0) as usize;
        let ctype = acc.get("componentType").and_then(Value::as_u64).unwrap_or(0) as u32;
        let ncomp = match acc.get("type").and_then(Value::as_str).unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => return err(format!("accessor {idx}: unsupported type {t}")),
        };
        let csize = match ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return err(format!("accessor {idx}: bad componentType {ctype}")),
        };
        let normalized = acc.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let Some(bv_idx) = acc.get("bufferView").and_then(Value::as_u64) else {
            // No bufferView: all zeros by spec.
            return Ok(Layout { count, ctype, ncomp, csize, normalized, start: 0, stride: 0, zero: true, bv: None });
        };
        let bv = self.arr("bufferViews").get(bv_idx as usize).ok_or_else(|| Error(format!("bufferView {bv_idx} missing")))?;
        if bv.get("buffer").and_then(Value::as_u64).unwrap_or(0) != 0 {
            return err("only the GLB-embedded buffer 0 is supported");
        }
        let bv_off = bv.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let bv_len = bv.get("byteLength").and_then(Value::as_u64).unwrap_or(0) as usize;
        let acc_off = acc.get("byteOffset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let elem = csize * ncomp;
        let stride = bv.get("byteStride").and_then(Value::as_u64).map(|s| s as usize).unwrap_or(elem);
        let start = bv_off + acc_off;
        if count > 0 && (start + stride * (count - 1) + elem > bv_off + bv_len || bv_off + bv_len > self.bin.len()) {
            return err(format!("accessor {idx} exceeds its buffer"));
        }
        Ok(Layout { count, ctype, ncomp, csize, normalized, start, stride, zero: false, bv: Some(bv_idx as usize) })
    }

    /// Reads any numeric accessor as f64 rows of `ncomp` values
    /// (normalized integers are mapped to [0,1] / [-1,1]).
    pub fn read_f64(&self, idx: usize) -> Result<(usize, Vec<f64>)> {
        let l = self.accessor_layout(idx)?;
        let mut out = Vec::with_capacity(l.count * l.ncomp);
        for i in 0..l.count {
            for c in 0..l.ncomp {
                if l.zero {
                    out.push(0.0);
                    continue;
                }
                let o = l.start + i * l.stride + c * l.csize;
                let b = &self.bin;
                let v = match l.ctype {
                    5120 => {
                        let x = b[o] as i8 as f64;
                        if l.normalized { (x / 127.0).max(-1.0) } else { x }
                    }
                    5121 => {
                        let x = b[o] as f64;
                        if l.normalized { x / 255.0 } else { x }
                    }
                    5122 => {
                        let x = i16::from_le_bytes([b[o], b[o + 1]]) as f64;
                        if l.normalized { (x / 32767.0).max(-1.0) } else { x }
                    }
                    5123 => {
                        let x = u16::from_le_bytes([b[o], b[o + 1]]) as f64;
                        if l.normalized { x / 65535.0 } else { x }
                    }
                    5125 => u32_at(b, o) as f64,
                    _ => f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as f64,
                };
                out.push(v);
            }
        }
        Ok((l.ncomp, out))
    }

    pub fn read_vec3(&self, idx: usize) -> Result<Vec<[f32; 3]>> {
        let (n, v) = self.read_f64(idx)?;
        if n != 3 {
            return err(format!("accessor {idx} is not VEC3"));
        }
        Ok(v.chunks(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect())
    }

    pub fn read_indices(&self, idx: usize) -> Result<Vec<u32>> {
        let (n, v) = self.read_f64(idx)?;
        if n != 1 {
            return err(format!("index accessor {idx} is not SCALAR"));
        }
        Ok(v.into_iter().map(|x| x as u32).collect())
    }

    /// Overwrites skinning data of a VEC4 JOINTS/WEIGHTS accessor in place.
    /// Returns false when the accessor's storage cannot hold the values
    /// (component type too small, zero-filled, shared bufferView we must not
    /// touch); the caller then appends a fresh accessor instead.
    pub fn overwrite_vec4(&mut self, idx: usize, rows: &[[f64; 4]]) -> Result<bool> {
        let l = self.accessor_layout(idx)?;
        if l.zero || l.ncomp != 4 || l.count != rows.len() {
            return Ok(false);
        }
        for (i, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                let o = l.start + i * l.stride + c * l.csize;
                let b = &mut self.bin;
                match l.ctype {
                    5121 => {
                        let x = if l.normalized { (v * 255.0).round() } else { v.round() };
                        if !(0.0..=255.0).contains(&x) {
                            return Ok(false);
                        }
                        b[o] = x as u8;
                    }
                    5123 => {
                        let x = if l.normalized { (v * 65535.0).round() } else { v.round() };
                        if !(0.0..=65535.0).contains(&x) {
                            return Ok(false);
                        }
                        b[o..o + 2].copy_from_slice(&(x as u16).to_le_bytes());
                    }
                    5126 => b[o..o + 4].copy_from_slice(&(v as f32).to_le_bytes()),
                    _ => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    pub fn component_type(&self, idx: usize) -> Option<(u32, bool)> {
        let a = self.arr("accessors").get(idx)?;
        Some((a.get("componentType")?.as_u64()? as u32, a.get("normalized").and_then(Value::as_bool).unwrap_or(false)))
    }

    /// Appends a tightly packed accessor and returns its index.
    /// `ctype` 5123 (u16) or 5126 (f32).
    pub fn append_vec4(&mut self, rows: &[[f64; 4]], ctype: u32) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let off = self.bin.len();
        for row in rows {
            for &v in row {
                match ctype {
                    5123 => self.bin.extend_from_slice(&(v.round().clamp(0.0, 65535.0) as u16).to_le_bytes()),
                    _ => self.bin.extend_from_slice(&(v as f32).to_le_bytes()),
                }
            }
        }
        let len = self.bin.len() - off;
        let obj = self.json.as_object_mut().expect("root object");
        let bvs = obj.entry("bufferViews").or_insert_with(|| json!([])).as_array_mut().unwrap();
        bvs.push(json!({"buffer": 0, "byteOffset": off, "byteLength": len}));
        let bv = bvs.len() - 1;
        let accs = obj.entry("accessors").or_insert_with(|| json!([])).as_array_mut().unwrap();
        accs.push(json!({"bufferView": bv, "componentType": ctype, "count": rows.len(), "type": "VEC4"}));
        let a = accs.len() - 1;
        self.sync_buffer_len();
        a
    }

    pub fn sync_buffer_len(&mut self) {
        let len = self.bin.len();
        let obj = self.json.as_object_mut().expect("root object");
        let bufs = obj.entry("buffers").or_insert_with(|| json!([{}])).as_array_mut().unwrap();
        if bufs.is_empty() {
            bufs.push(json!({}));
        }
        bufs[0]["byteLength"] = json!(len);
    }

    /// Appends raw bytes as a bufferView (4-byte aligned) and returns its index.
    pub fn append_view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let off = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let obj = self.json.as_object_mut().expect("root object");
        let bvs = obj.entry("bufferViews").or_insert_with(|| json!([])).as_array_mut().unwrap();
        let mut bv = json!({"buffer": 0, "byteOffset": off, "byteLength": bytes.len()});
        if let Some(t) = target {
            bv["target"] = json!(t);
        }
        bvs.push(bv);
        let i = bvs.len() - 1;
        self.sync_buffer_len();
        i
    }

    pub fn push_accessor(&mut self, acc: Value) -> usize {
        let obj = self.json.as_object_mut().expect("root object");
        let accs = obj.entry("accessors").or_insert_with(|| json!([])).as_array_mut().unwrap();
        accs.push(acc);
        accs.len() - 1
    }
}

struct Layout {
    count: usize,
    ctype: u32,
    ncomp: usize,
    csize: usize,
    normalized: bool,
    start: usize,
    stride: usize,
    zero: bool,
    #[allow(dead_code)]
    bv: Option<usize>,
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
