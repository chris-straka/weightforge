//! In-process software rasterizer for pose sheets: deterministic (golden
//! images), headless, no GPU or Blender needed. Orthographic 3/4 view,
//! two-sided Lambert shading (AI meshes often have flipped normals), 2x
//! supersampling, per-vertex colors with bad vertices in red.

use crate::font::{draw_text, text_width};
use crate::math::{Mat4, Vec3, v3};
use crate::metrics::{Ctx, Eval, F_FOLLOW, F_INTERSECT};
use crate::scene::{Model, Weights};
use crate::skin::lbs;

pub struct Image {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

pub const BG: [u8; 3] = [24, 26, 31];
const FG: [u8; 3] = [232, 234, 238];
const DIM: [u8; 3] = [150, 156, 168];
const RED: [u8; 3] = [235, 64, 52];
const GREEN: [u8; 3] = [92, 200, 120];
const BODY: [f64; 3] = [0.80, 0.80, 0.83];
const PIECE: [f64; 3] = [0.58, 0.68, 0.86];
const BAD: [f64; 3] = [0.95, 0.16, 0.10];
const ISECT: [f64; 3] = [0.98, 0.62, 0.18];

impl Image {
    pub fn new(w: usize, h: usize, bg: [u8; 3]) -> Image {
        let mut rgb = vec![0u8; w * h * 3];
        for px in rgb.chunks_mut(3) {
            px.copy_from_slice(&bg);
        }
        Image { w, h, rgb }
    }
    pub fn rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: [u8; 3]) {
        for yy in y..(y + h).min(self.h) {
            for xx in x..(x + w).min(self.w) {
                let o = (yy * self.w + xx) * 3;
                self.rgb[o..o + 3].copy_from_slice(&c);
            }
        }
    }
    pub fn text(&mut self, x: usize, y: usize, s: &str, scale: usize, c: [u8; 3]) {
        draw_text(&mut self.rgb, self.w, self.h, x, y, s, scale, c);
    }
    pub fn png_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, self.w as u32, self.h as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_compression(png::Compression::Balanced);
            let mut wr = enc.write_header().expect("png header");
            wr.write_image_data(&self.rgb).expect("png data");
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct View {
    pub yaw_deg: f64,
    pub pitch_deg: f64,
}

impl Default for View {
    fn default() -> Self {
        View { yaw_deg: 30.0, pitch_deg: 8.0 }
    }
}

impl View {
    /// Camera basis (right, up, toward viewer) for a camera in front of a
    /// +Z-forward character, turned by yaw about +Y and tilted by pitch.
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw_deg.to_radians().sin_cos();
        let (sp, cp) = self.pitch_deg.to_radians().sin_cos();
        let back = v3(sy * cp, sp, cy * cp); // from target toward camera
        let right = v3(0.0, 1.0, 0.0).cross(back).normalized();
        let up = back.cross(right).normalized();
        (right, up, back)
    }
    pub fn to_cam(&self, p: Vec3) -> Vec3 {
        let (r, u, b) = self.basis();
        v3(p.dot(r), p.dot(u), p.dot(b))
    }
}

/// Screen fit shared by every cell of a sheet: camera-space center and
/// pixels per unit.
#[derive(Clone, Copy)]
pub struct Fit {
    pub cx: f64,
    pub cy: f64,
    pub ppu: f64,
}

pub fn fit(view: &View, sets: &[&[Vec3]], cell_w: usize, cell_h: usize) -> Fit {
    let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, 0.0), v3(f64::MIN, f64::MIN, 0.0));
    for s in sets {
        for p in s.iter() {
            let q = view.to_cam(*p);
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    let w = (hi.x - lo.x).max(1e-9);
    let h = (hi.y - lo.y).max(1e-9);
    let ppu = 0.92 * (cell_w as f64 / w).min(cell_h as f64 / h);
    Fit { cx: (lo.x + hi.x) * 0.5, cy: (lo.y + hi.y) * 0.5, ppu }
}

/// Rasterizes a mesh into `img` inside the cell (x0, y0, w, h).
#[allow(clippy::too_many_arguments)]
pub fn draw_mesh(
    img: &mut Image,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    pos: &[Vec3],
    tris: &[[u32; 3]],
    colors: &[[f64; 3]],
    view: &View,
    f: Fit,
) {
    const SS: usize = 2;
    let (sw, sh) = (w * SS, h * SS);
    let mut color = vec![[0.0f64; 3]; sw * sh];
    let mut depth = vec![f64::NEG_INFINITY; sw * sh];
    let cam: Vec<Vec3> = pos.iter().map(|p| view.to_cam(*p)).collect();
    let scr: Vec<(f64, f64, f64)> = cam
        .iter()
        .map(|q| (((q.x - f.cx) * f.ppu + w as f64 * 0.5) * SS as f64, ((f.cy - q.y) * f.ppu + h as f64 * 0.5) * SS as f64, q.z))
        .collect();
    let light = v3(-0.35, 0.55, 1.0).normalized();
    for t in tris {
        let (a, b, c) = (scr[t[0] as usize], scr[t[1] as usize], scr[t[2] as usize]);
        let n = (cam[t[1] as usize] - cam[t[0] as usize]).cross(cam[t[2] as usize] - cam[t[0] as usize]).normalized();
        let shade = 0.28 + 0.72 * n.dot(light).abs();
        let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area.abs() < 1e-12 {
            continue;
        }
        let minx = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
        let maxx = (a.0.max(b.0).max(c.0).ceil() as isize).min(sw as isize - 1);
        let miny = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
        let maxy = (a.1.max(b.1).max(c.1).ceil() as isize).min(sh as isize - 1);
        if maxx < 0 || maxy < 0 {
            continue;
        }
        let (ca, cb, cc) = (colors[t[0] as usize], colors[t[1] as usize], colors[t[2] as usize]);
        for py in miny..=maxy as usize {
            for px in minx..=maxx as usize {
                let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
                let w0 = ((b.0 - x) * (c.1 - y) - (b.1 - y) * (c.0 - x)) / area;
                let w1 = ((c.0 - x) * (a.1 - y) - (c.1 - y) * (a.0 - x)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a.2 + w1 * b.2 + w2 * c.2;
                let i = py * sw + px;
                if z <= depth[i] {
                    continue;
                }
                depth[i] = z;
                for k in 0..3 {
                    color[i][k] = (ca[k] * w0 + cb[k] * w1 + cc[k] * w2) * shade;
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f64; 3];
            let mut cov = 0.0;
            for dy in 0..SS {
                for dx in 0..SS {
                    let i = (y * SS + dy) * sw + x * SS + dx;
                    if depth[i] > f64::NEG_INFINITY {
                        cov += 1.0;
                        for k in 0..3 {
                            acc[k] += color[i][k];
                        }
                    }
                }
            }
            if cov == 0.0 {
                continue;
            }
            let o = ((y0 + y) * img.w + x0 + x) * 3;
            let n = (SS * SS) as f64;
            for k in 0..3 {
                let bg = img.rgb[o + k] as f64 / 255.0;
                let v = acc[k] / n + bg * (1.0 - cov / n);
                img.rgb[o + k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
}

pub struct SheetOpts {
    pub cols: usize,
    pub cell_w: usize,
    pub cell_h: usize,
    /// Poses shown besides rest (worst first).
    pub max_poses: usize,
    pub view: View,
}

impl Default for SheetOpts {
    fn default() -> Self {
        SheetOpts { cols: 4, cell_w: 300, cell_h: 380, max_poses: 11, view: View::default() }
    }
}

const HEADER: usize = 96;
const LABEL: usize = 30;

fn base_colors(model: &Model) -> Vec<[f64; 3]> {
    (0..model.nverts()).map(|v| if model.parts[model.vpart[v] as usize].piece { PIECE } else { BODY }).collect()
}

/// Paints a pose's reported vertices; returns how many fail (not just
/// intersect).
fn paint(model: &Model, colors: &mut [[f64; 3]], bad: &[(u32, u8)], mask: &[u8]) -> usize {
    let mut n = 0;
    // A piece that leaves the body leaves as a whole: paint all of it.
    let mut torn = vec![false; model.parts.len()];
    for &(v, f) in bad {
        if f & mask[v as usize] & F_FOLLOW != 0 {
            torn[model.vpart[v as usize] as usize] = true;
        }
    }
    for (v, c) in colors.iter_mut().enumerate() {
        if torn[model.vpart[v] as usize] {
            *c = BAD;
        }
    }
    for &(v, f) in bad {
        let f = f & mask[v as usize];
        if f & !F_INTERSECT != 0 {
            colors[v as usize] = BAD;
            n += 1;
        } else if f != 0 {
            colors[v as usize] = ISECT;
        }
    }
    n
}

fn masked_pose_bad(ev: &Eval, mask: &[u8]) -> Vec<usize> {
    ev.bad_in_pose.iter().map(|l| l.iter().filter(|(v, f)| f & mask[*v as usize] & !F_INTERSECT != 0).count()).collect()
}

/// Worst poses first (most bad vertices), at most two per moved bone so
/// one bad joint does not fill the sheet, then ROM order to fill it.
pub fn pick_poses(ctx: &Ctx, bad: &[usize], max: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..ctx.poses.len()).collect();
    idx.sort_by(|&a, &b| bad[b].cmp(&bad[a]).then(a.cmp(&b)));
    let key = |p: usize| ctx.poses[p].moves.first().map(|m| m.0 as i64).unwrap_or(-1 - p as i64);
    let mut per: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let mut out = Vec::new();
    let mut rest = Vec::new();
    for p in idx {
        let c = per.entry(key(p)).or_default();
        if *c < 2 {
            *c += 1;
            out.push(p);
        } else {
            rest.push(p);
        }
    }
    out.extend(rest);
    out.truncate(max);
    out
}

fn posed(model: &Model, ctx: &Ctx, p: Option<usize>, w: &Weights) -> Vec<Vec3> {
    match p {
        Some(p) => lbs(model, &ctx.mats[p], w),
        None => {
            let id = vec![Mat4::IDENTITY; model.njoints()];
            lbs(model, &id, w)
        }
    }
}

fn verdict_line(score: f64, pass: bool, fails: usize) -> (String, [u8; 3]) {
    if pass { (format!("SCORE {score:.1}/100  PASS"), GREEN) } else { (format!("SCORE {score:.1}/100  FAIL  {fails} FINDINGS"), RED) }
}

pub struct SheetInfo<'a> {
    pub title: &'a str,
    /// Per-vertex flags to paint (see `report::finding_mask`).
    pub mask: &'a [u8],
    pub score: f64,
    pub pass: bool,
    pub fails: usize,
}

/// Range-of-motion sheet: rest + the worst poses, bad vertices in red.
pub fn sheet(model: &Model, ctx: &Ctx, ev: &Eval, w: &Weights, info: &SheetInfo, o: &SheetOpts) -> Image {
    let pose_bad = masked_pose_bad(ev, info.mask);
    let poses = pick_poses(ctx, &pose_bad, o.max_poses);
    let mut cells: Vec<Option<usize>> = vec![None];
    cells.extend(poses.iter().map(|&p| Some(p)));
    let meshes: Vec<Vec<Vec3>> = cells.iter().map(|&p| posed(model, ctx, p, w)).collect();
    let refs: Vec<&[Vec3]> = meshes.iter().map(|m| m.as_slice()).collect();
    let f = fit(&o.view, &refs, o.cell_w, o.cell_h - LABEL);
    let rows = cells.len().div_ceil(o.cols);
    let mut img = Image::new(o.cols * o.cell_w, HEADER + rows * o.cell_h, BG);
    img.text(16, 14, &format!("WEIGHTFORGE  {}", info.title), 3, FG);
    let (line, col) = verdict_line(info.score, info.pass, info.fails);
    img.text(16, 50, &line, 3, col);
    let legend = "RED = BAD IN THAT POSE";
    img.text(img.w.saturating_sub(text_width(legend, 2) + 16), 46, legend, 2, DIM);
    let legend2 = "ORANGE = PASSES THROUGH BODY";
    img.text(img.w.saturating_sub(text_width(legend2, 2) + 16), 68, legend2, 2, [250, 158, 46]);
    let base = base_colors(model);
    for (k, &p) in cells.iter().enumerate() {
        let (cx, cy) = ((k % o.cols) * o.cell_w, HEADER + (k / o.cols) * o.cell_h);
        img.rect(cx + 2, cy + 2, o.cell_w - 4, o.cell_h - 4, [32, 35, 42]);
        let mut colors = base.clone();
        let (label, n) = match p {
            Some(p) => {
                let n = paint(model, &mut colors, &ev.bad_in_pose[p], info.mask);
                (ctx.poses[p].name.clone(), n)
            }
            None => ("REST".to_string(), 0),
        };
        let lab = if p.is_some() { format!("{label} ({n})") } else { label };
        img.text(cx + 10, cy + 9, &lab, 2, if n > 0 { RED } else { FG });
        draw_mesh(&mut img, cx + 2, cy + LABEL, o.cell_w - 4, o.cell_h - LABEL - 4, &meshes[k], &model.tris, &colors, &o.view, f);
    }
    img
}

pub struct Side<'a> {
    pub model: &'a Model,
    pub ctx: &'a Ctx,
    pub ev: &'a Eval,
    pub w: &'a Weights,
    pub info: SheetInfo<'a>,
}

/// A/B sheet: same poses side by side, one pose per row.
pub fn compare(a: &Side, b: &Side, o: &SheetOpts) -> Image {
    let names_b: Vec<&str> = b.ctx.poses.iter().map(|p| p.name.as_str()).collect();
    let mut bad = masked_pose_bad(a.ev, a.info.mask);
    let bad_b = masked_pose_bad(b.ev, b.info.mask);
    for (i, p) in a.ctx.poses.iter().enumerate() {
        if let Some(j) = names_b.iter().position(|n| *n == p.name) {
            bad[i] += bad_b[j];
        }
    }
    let picked = pick_poses(a.ctx, &bad, o.max_poses.min(7));
    let mut rows: Vec<(Option<usize>, Option<usize>)> = vec![(None, None)];
    for p in picked {
        if let Some(j) = names_b.iter().position(|n| *n == a.ctx.poses[p].name) {
            rows.push((Some(p), Some(j)));
        }
    }
    let ma: Vec<Vec<Vec3>> = rows.iter().map(|r| posed(a.model, a.ctx, r.0, a.w)).collect();
    let mb: Vec<Vec<Vec3>> = rows.iter().map(|r| posed(b.model, b.ctx, r.1, b.w)).collect();
    let refs: Vec<&[Vec3]> = ma.iter().chain(mb.iter()).map(|m| m.as_slice()).collect();
    let f = fit(&o.view, &refs, o.cell_w, o.cell_h - LABEL);
    let header = HEADER + 40;
    let mut img = Image::new(2 * o.cell_w, header + rows.len() * o.cell_h, BG);
    for (k, side) in [a, b].iter().enumerate() {
        let x = 12 + k * o.cell_w;
        img.text(x, 12, if k == 0 { "A" } else { "B" }, 4, FG);
        img.text(x + 36, 18, side.info.title, 2, FG);
        let (line, col) = verdict_line(side.info.score, side.info.pass, side.info.fails);
        img.text(x, 56, &line, 2, col);
    }
    img.text(12, HEADER + 8, "SAME POSES. RED = BAD, ORANGE = PASSES THROUGH", 2, DIM);
    for (r, (pa, pb)) in rows.iter().enumerate() {
        let cy = header + r * o.cell_h;
        for (k, (side, p, mesh)) in [(a, pa, &ma[r]), (b, pb, &mb[r])].into_iter().enumerate() {
            let cx = k * o.cell_w;
            img.rect(cx + 2, cy + 2, o.cell_w - 4, o.cell_h - 4, [32, 35, 42]);
            let mut colors = base_colors(side.model);
            let (label, n) = match p {
                Some(p) => {
                    let n = paint(side.model, &mut colors, &side.ev.bad_in_pose[*p], side.info.mask);
                    (side.ctx.poses[*p].name.clone(), n)
                }
                None => ("REST".into(), 0),
            };
            let lab = if p.is_some() { format!("{label} ({n})") } else { label };
            img.text(cx + 10, cy + 9, &lab, 2, if n > 0 { RED } else { FG });
            draw_mesh(&mut img, cx + 2, cy + LABEL, o.cell_w - 4, o.cell_h - LABEL - 4, mesh, &side.model.tris, &colors, &o.view, f);
        }
    }
    img
}
