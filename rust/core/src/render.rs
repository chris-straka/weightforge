//! In-process software rasterizer for pose sheets: deterministic (golden
//! images), headless, no GPU or Blender needed. Orthographic 3/4 view,
//! two-sided Lambert shading (AI meshes often have flipped normals), 2x
//! supersampling, per-vertex colors with bad vertices in red.

use crate::font::{draw_text, text_width};
use crate::math::{Mat4, Vec3, v3};
use crate::metrics::{Ctx, Eval, F_FOLLOW, F_INTERSECT};
use crate::scene::{Model, Weights};
use crate::skin::{joint_matrices, lbs, pose_at};

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

/// Shortens `s` to `max` glyphs by cutting the middle ("LONG_NA..E.GLB").
pub fn fit_text(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 5 {
        return s.to_string();
    }
    let keep = max - 2;
    let head = keep / 2;
    let tail = keep - head;
    chars[..head].iter().chain(['.', '.'].iter()).chain(chars[chars.len() - tail..].iter()).collect()
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
    let title_cols = (o.cols * o.cell_w).saturating_sub(32) / 18;
    img.text(16, 14, &fit_text(&format!("WEIGHTFORGE  {}", info.title), title_cols), 3, FG);
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
    // Each side's header must fit its own column (12 px per glyph at scale 2).
    let cols = (o.cell_w.saturating_sub(60)) / 12;
    for (k, side) in [a, b].iter().enumerate() {
        let x = 12 + k * o.cell_w;
        img.text(x, 12, if k == 0 { "A" } else { "B" }, 4, FG);
        img.text(x + 36, 18, &fit_text(side.info.title, cols), 2, FG);
        let (line, col) = if side.info.pass {
            (format!("{:.1}/100 PASS", side.info.score), GREEN)
        } else {
            (format!("{:.1}/100 FAIL ({})", side.info.score, side.info.fails), RED)
        };
        img.text(x, 56, &fit_text(&line, cols + 3), 2, col);
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

// ---- Zoomed compare: one pose a row, the worst first, cropped to the fault ----

/// One row of the zoomed compare: a pose by name, and for each rig its
/// pose index and failing-vertex count (None: that rig has no such pose).
#[derive(Clone, Debug, PartialEq)]
pub struct ZoomRow {
    pub name: String,
    pub a: Option<(usize, usize)>,
    pub b: Option<(usize, usize)>,
}

impl ZoomRow {
    fn counts(&self) -> (usize, usize) {
        (self.a.map_or(0, |x| x.1), self.b.map_or(0, |x| x.1))
    }
}

/// Poses matched by name and ranked by the larger of the two rigs' fault
/// counts (then by their sum, then range-of-motion order). A pose only one
/// rig has is kept; a pose neither rig fails is left out.
pub fn rank_rows(names_a: &[&str], bad_a: &[usize], names_b: &[&str], bad_b: &[usize], max: usize) -> Vec<ZoomRow> {
    let mut rows: Vec<ZoomRow> = names_a
        .iter()
        .enumerate()
        .map(|(i, n)| ZoomRow {
            name: n.to_string(),
            a: Some((i, bad_a[i])),
            b: names_b.iter().position(|m| m == n).map(|j| (j, bad_b[j])),
        })
        .collect();
    for (j, n) in names_b.iter().enumerate() {
        if !names_a.contains(n) {
            rows.push(ZoomRow { name: n.to_string(), a: None, b: Some((j, bad_b[j])) });
        }
    }
    rows.retain(|r| r.counts() != (0, 0));
    rows.sort_by(|x, y| {
        let ((xa, xb), (ya, yb)) = (x.counts(), y.counts());
        ya.max(yb).cmp(&xa.max(xb)).then((ya + yb).cmp(&(xa + xb)))
    });
    rows.truncate(max);
    rows
}

pub struct ZoomOpts<'a> {
    /// Poses shown (the worst first).
    pub rows: usize,
    pub cell_w: usize,
    pub cell_h: usize,
    pub label_a: &'a str,
    pub label_b: &'a str,
    /// A line under each rig's name (its worst part, say).
    pub note_a: &'a str,
    pub note_b: &'a str,
}

impl Default for ZoomOpts<'_> {
    fn default() -> Self {
        ZoomOpts { rows: 5, cell_w: 520, cell_h: 420, label_a: "A", label_b: "B", note_a: "", note_b: "" }
    }
}

const ZHEADER: usize = 186;
const ZROW: usize = 62;
const CLAY: [f64; 3] = [0.62, 0.62, 0.64];
const HOT: [f64; 3] = [1.0, 0.08, 0.05];
const WARM: [f64; 3] = [1.0, 0.80, 0.15];
const BOX: [u8; 3] = [255, 214, 64];
/// The smallest crop, in body heights: a fault is shown with its joint
/// around it, never as a few triangles filling the cell.
const ZOOM_MIN: f64 = 0.30;

/// The smallest crop of a probe case: wider than a compare row's, since
/// nothing is marked and the eye needs the whole joint to judge it.
const PROBE_MIN: f64 = 0.5;

/// Frames and rate of a row's loop (`compare_zoom_clips`): two seconds.
pub const CLIP_FRAMES: usize = 60;
pub const CLIP_FPS: usize = 30;

/// Cameras tried for each row; the one that shows the most of both rigs'
/// failing vertices is used for both.
const ZOOM_VIEWS: &[(f64, f64, &str)] = &[
    (30.0, 8.0, "FRONT LEFT"),
    (-30.0, 8.0, "FRONT RIGHT"),
    (150.0, 8.0, "BACK LEFT"),
    (-150.0, 8.0, "BACK RIGHT"),
    (30.0, -40.0, "FRONT LEFT, FROM BELOW"),
    (-30.0, -40.0, "FRONT RIGHT, FROM BELOW"),
    (150.0, -40.0, "BACK LEFT, FROM BELOW"),
    (-150.0, -40.0, "BACK RIGHT, FROM BELOW"),
    (30.0, 40.0, "FRONT LEFT, FROM ABOVE"),
    (-30.0, 40.0, "FRONT RIGHT, FROM ABOVE"),
    (150.0, 40.0, "BACK LEFT, FROM ABOVE"),
    (-150.0, 40.0, "BACK RIGHT, FROM ABOVE"),
];

/// Vertices to paint in a pose: the failing ones, and all of a piece that
/// tears off.
fn failing(model: &Model, bad: &[(u32, u8)], mask: &[u8]) -> Vec<bool> {
    let mut torn = vec![false; model.parts.len()];
    let mut out = vec![false; model.nverts()];
    for &(v, f) in bad {
        let f = f & mask[v as usize];
        if f & F_FOLLOW != 0 {
            torn[model.vpart[v as usize] as usize] = true;
        }
        if f & !F_INTERSECT != 0 {
            out[v as usize] = true;
        }
    }
    for (v, o) in out.iter_mut().enumerate() {
        *o |= torn[model.vpart[v] as usize];
    }
    out
}

/// A rig's rest body in a camera: centre line, feet and height. Crops are
/// placed in these units so two rigs of different size frame the same
/// part of the body.
#[derive(Clone, Copy)]
struct Frame {
    cx: f64,
    y0: f64,
    h: f64,
}

fn frame(view: &View, rest: &[Vec3]) -> Frame {
    let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, 0.0), v3(f64::MIN, f64::MIN, 0.0));
    for p in rest {
        let q = view.to_cam(*p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    Frame { cx: (lo.x + hi.x) * 0.5, y0: lo.y, h: (hi.y - lo.y).max(1e-9) }
}

/// The 4th and 96th percentile of `v` (its whole range when short), so
/// one stray vertex does not widen the crop.
fn span(v: &mut [f64]) -> (f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    let cut = if v.len() >= 25 { v.len() / 25 } else { 0 };
    (v[cut], v[v.len() - 1 - cut])
}

/// One crop for both rigs: the box around their failing vertices in this
/// pose, in body units, padded and held to the cell's shape. None when
/// neither has one.
fn zoom_fits(view: &View, rest: [&[Vec3]; 2], mesh: [&[Vec3]; 2], flags: [&[bool]; 2], cw: usize, ch: usize, min: f64) -> Option<[Fit; 2]> {
    let frames = [frame(view, rest[0]), frame(view, rest[1])];
    let (mut us, mut vs) = (Vec::new(), Vec::new());
    for k in 0..2 {
        for (p, _) in mesh[k].iter().zip(flags[k]).filter(|(_, f)| **f) {
            let q = view.to_cam(*p);
            us.push((q.x - frames[k].cx) / frames[k].h);
            vs.push((q.y - frames[k].y0) / frames[k].h);
        }
    }
    if us.is_empty() {
        return None;
    }
    let ((u0, u1), (v0, v1)) = (span(&mut us), span(&mut vs));
    let aspect = cw as f64 / ch as f64;
    let (bw, bh) = ((u1 - u0) * 1.4 + 0.06, (v1 - v0) * 1.4 + 0.06);
    let hh = bh.max(bw / aspect).max(min);
    let (uc, vc) = ((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    Some(frames.map(|fr| Fit { cx: fr.cx + uc * fr.h, cy: fr.y0 + vc * fr.h, ppu: ch as f64 / (hh * fr.h) }))
}

/// Depth-tested triangle fill over screen-space vertices (x, y in pixels,
/// z toward the viewer): `hit(triangle, pixel, barycentrics)` is called for
/// every sample that is the nearest so far. Returns the depth buffer.
fn raster(scr: &[(f64, f64, f64)], tris: &[[u32; 3]], w: usize, h: usize, mut hit: impl FnMut(usize, usize, [f64; 3])) -> Vec<f64> {
    let mut depth = vec![f64::NEG_INFINITY; w * h];
    for (ti, t) in tris.iter().enumerate() {
        let (a, b, c) = (scr[t[0] as usize], scr[t[1] as usize], scr[t[2] as usize]);
        let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        if area.abs() < 1e-12 {
            continue;
        }
        let minx = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
        let maxx = (a.0.max(b.0).max(c.0).ceil() as isize).min(w as isize - 1);
        let miny = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
        let maxy = (a.1.max(b.1).max(c.1).ceil() as isize).min(h as isize - 1);
        if maxx < 0 || maxy < 0 {
            continue;
        }
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
                let i = py * w + px;
                if z > depth[i] {
                    depth[i] = z;
                    hit(ti, i, [w0, w1, w2]);
                }
            }
        }
    }
    depth
}

/// How many marked vertices a camera sees inside a w x h cell: in frame
/// and not behind the body.
fn visible(pos: &[Vec3], tris: &[[u32; 3]], marked: &[bool], view: &View, f: Fit, w: usize, h: usize) -> usize {
    let scr: Vec<(f64, f64, f64)> = pos
        .iter()
        .map(|p| {
            let q = view.to_cam(*p);
            ((q.x - f.cx) * f.ppu + w as f64 * 0.5, (f.cy - q.y) * f.ppu + h as f64 * 0.5, q.z)
        })
        .collect();
    let depth = raster(&scr, tris, w, h, |_, _, _| {});
    // A vertex counts as seen when nothing is more than 2% of the crop's
    // height in front of it.
    let eps = 0.02 * h as f64 / f.ppu;
    let mut n = 0;
    for (s, m) in scr.iter().zip(marked) {
        if !*m || s.0 < 0.0 || s.1 < 0.0 || s.0 >= w as f64 || s.1 >= h as f64 {
            continue;
        }
        if s.2 >= depth[s.1 as usize * w + s.0 as usize] - eps {
            n += 1;
        }
    }
    n
}

/// Area-weighted vertex normals (not normalized).
fn vertex_normals(pos: &[Vec3], tris: &[[u32; 3]]) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; pos.len()];
    for t in tris {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let f = (b - a).cross(c - a);
        for &i in t {
            n[i as usize] += f;
        }
    }
    n
}

/// Clay render for close-ups: smooth normals under one strong light from
/// the side, so stretched and collapsed skin reads as creases and dents.
/// Each face is one flat colour, `face[t] = (colour, lift)`; lift (0 to 1)
/// raises a marked face out of the shading so it stays bright in shadow.
#[allow(clippy::too_many_arguments)]
fn draw_clay(
    img: &mut Image,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    pos: &[Vec3],
    tris: &[[u32; 3]],
    face: &[([f64; 3], f64)],
    view: &View,
    f: Fit,
) {
    const SS: usize = 2;
    let (sw, sh) = (w * SS, h * SS);
    let cam: Vec<Vec3> = pos.iter().map(|p| view.to_cam(*p)).collect();
    let scr: Vec<(f64, f64, f64)> = cam
        .iter()
        .map(|q| (((q.x - f.cx) * f.ppu + w as f64 * 0.5) * SS as f64, ((f.cy - q.y) * f.ppu + h as f64 * 0.5) * SS as f64, q.z))
        .collect();
    let normals = vertex_normals(&cam, tris);
    // From the viewer's left, above, a little in front.
    let key = v3(-0.80, 0.45, 0.40).normalized();
    let mut color = vec![[0.0f64; 3]; sw * sh];
    let depth = raster(&scr, tris, sw, sh, |t, i, b| {
        let tri = tris[t];
        let mut n = (normals[tri[0] as usize] * b[0] + normals[tri[1] as usize] * b[1] + normals[tri[2] as usize] * b[2]).normalized();
        if n.z < 0.0 {
            n = -n;
        }
        let shade = (0.24 + 0.14 * n.z + 0.72 * n.dot(key).max(0.0)).min(1.0);
        let (c, lift) = face[t];
        let s = shade + (1.0 - shade) * lift;
        color[i] = [c[0] * s, c[1] * s, c[2] * s];
    });
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

fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Face colours for a pose: a face with a failing vertex is filled solid
/// red (`amount` 1) or part of the way there (a clip easing in).
fn fault_faces(tris: &[[u32; 3]], flags: &[bool], amount: f64) -> Vec<([f64; 3], f64)> {
    tris.iter().map(|t| if t.iter().any(|&v| flags[v as usize]) { (mix(CLAY, HOT, amount), 0.8 * amount) } else { (CLAY, 0.0) }).collect()
}

/// Face colours for the "where the score is lost" row: grey through
/// yellow to red by the worst deformation energy among a face's vertices,
/// on a fixed scale (1.0 and above is full red) so two rigs compare.
fn heat_faces(tris: &[[u32; 3]], energy: &[f64]) -> Vec<([f64; 3], f64)> {
    tris.iter()
        .map(|t| {
            let e = t.iter().map(|&v| energy[v as usize]).fold(0.0, f64::max);
            let v = e.clamp(0.0, 1.0).sqrt();
            let c = if v < 0.5 { mix(CLAY, WARM, v * 2.0) } else { mix(WARM, HOT, v * 2.0 - 1.0) };
            (c, 0.7 * v)
        })
        .collect()
}

/// The line that says which rig is better and by how much, and the
/// numbers behind it. Weightforge ranks rigs by the gate first, then the
/// score (as `fix` ranks its candidates); the higher score with more
/// failing findings is called mixed.
pub fn zoom_headline(a: &SheetInfo, b: &SheetInfo, label_a: &str, label_b: &str, bad_a: usize, bad_b: usize) -> (String, String) {
    let b_wins = if a.pass != b.pass { b.pass } else { b.score > a.score };
    let ((w, wl, wb), (l, ll, lb)) =
        if b_wins { ((b, label_b, bad_b), (a, label_a, bad_a)) } else { ((a, label_a, bad_a), (b, label_b, bad_b)) };
    let head = if w.pass != l.pass {
        format!("{wl} PASSES THE GATE, {ll} DOES NOT")
    } else if (w.score - l.score).abs() < 1.0 {
        format!("ABOUT EVEN: {wl} {:.1}, {ll} {:.1} OUT OF 100", w.score, l.score)
    } else if w.fails > l.fails {
        // The higher score with more failing findings is not a clear win.
        format!("MIXED: {wl} SCORES {:.1} VS {:.1} BUT FAILS MORE", w.score, l.score)
    } else {
        format!("{wl} IS BETTER: {:.1} VS {:.1} OUT OF 100", w.score, l.score)
    };
    let detail = format!("{wl} VS {ll}: FAILING FINDINGS {} VS {}, BAD VERTS OVER ALL POSES {} VS {}", w.fails, l.fails, wb, lb);
    (head, detail)
}

fn outline(img: &mut Image, x0: usize, y0: usize, x1: usize, y1: usize, c: [u8; 3]) {
    let (w, h) = (x1.saturating_sub(x0).max(2), y1.saturating_sub(y0).max(2));
    img.rect(x0, y0, w, 2, c);
    img.rect(x0, y0 + h - 2, w, 2, c);
    img.rect(x0, y0, 2, h, c);
    img.rect(x0 + w - 2, y0, 2, h, c);
}

/// One row, ready to draw: its pose on each rig, the camera, each rig's
/// crop and its failing vertices.
struct ZoomView {
    row: ZoomRow,
    view: View,
    view_name: &'static str,
    fits: [Fit; 2],
    flags: [Vec<bool>; 2],
}

/// A rig's joint matrices `t` of the way from rest (0) into a pose (1).
fn mats_at(s: &Side, pick: Option<(usize, usize)>, t: f64) -> Vec<Mat4> {
    match pick {
        Some((p, _)) if t >= 1.0 => s.ctx.mats[p].clone(),
        Some((p, _)) if t > 0.0 => joint_matrices(s.model, &pose_at(s.model, &s.ctx.poses[p], t)),
        _ => vec![Mat4::IDENTITY; s.model.njoints()],
    }
}

/// A rig's mesh `t` of the way from rest (0) into a pose (1).
fn mesh_at(s: &Side, pick: Option<(usize, usize)>, t: f64) -> Vec<Vec3> {
    lbs(s.model, &mats_at(s, pick, t), s.w)
}

/// Every bone's posed head and tail, for drawing the skeleton over the
/// body.
fn bones_at(s: &Side, mats: &[Mat4]) -> Vec<(Vec3, Vec3)> {
    let sk = &s.model.skel;
    (0..s.model.njoints()).map(|j| (mats[j].transform_point(sk.head[j]), mats[j].transform_point(sk.tail[j]))).collect()
}

const BONE: [u8; 3] = [70, 200, 255];
const JOINT: [u8; 3] = [255, 214, 64];

/// A filled disc inside the cell (x0, y0, w, h), clipped to it.
#[allow(clippy::too_many_arguments)]
fn disc(img: &mut Image, x0: usize, y0: usize, w: usize, h: usize, x: f64, y: f64, r: f64, c: [u8; 3]) {
    if x + r < 0.0 || y + r < 0.0 || x - r >= w as f64 || y - r >= h as f64 {
        return;
    }
    let (lo_x, hi_x) = ((x - r).floor().max(0.0) as usize, ((x + r).ceil() as usize).min(w.saturating_sub(1)));
    let (lo_y, hi_y) = ((y - r).floor().max(0.0) as usize, ((y + r).ceil() as usize).min(h.saturating_sub(1)));
    for py in lo_y..=hi_y {
        for px in lo_x..=hi_x {
            let (dx, dy) = (px as f64 + 0.5 - x, py as f64 + 0.5 - y);
            if dx * dx + dy * dy <= r * r {
                let o = ((y0 + py) * img.w + x0 + px) * 3;
                img.rgb[o..o + 3].copy_from_slice(&c);
            }
        }
    }
}

/// The skeleton drawn through the body, inside the cell (x0, y0, w, h):
/// a line along each bone and a dot at each joint, so the eye can lay the
/// skin against where it is meant to bend.
#[allow(clippy::too_many_arguments)]
fn draw_bones(img: &mut Image, x0: usize, y0: usize, w: usize, h: usize, bones: &[(Vec3, Vec3)], view: &View, f: Fit) {
    let to = |p: Vec3| {
        let q = view.to_cam(p);
        ((q.x - f.cx) * f.ppu + w as f64 * 0.5, (f.cy - q.y) * f.ppu + h as f64 * 0.5)
    };
    for &(head, tail) in bones {
        let (a, b) = (to(head), to(tail));
        let len = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        // Walked a pixel at a time; `disc` clips what falls outside. A
        // bone far longer than the cell is one that starts far outside it.
        if len > 6.0 * (w + h) as f64 {
            continue;
        }
        let steps = len.ceil().max(1.0) as usize;
        for k in 0..=steps {
            let t = k as f64 / steps as f64;
            disc(img, x0, y0, w, h, a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, 1.3, BONE);
        }
    }
    for &(head, _) in bones {
        let a = to(head);
        disc(img, x0, y0, w, h, a.0, a.1, 5.5, [20, 22, 26]);
        disc(img, x0, y0, w, h, a.0, a.1, 4.0, JOINT);
    }
}

/// The rows of the zoomed compare, worst first, each with the camera that
/// shows the most of both rigs' failing vertices and the crop around them.
fn zoom_views(sides: [&Side; 2], o: &ZoomOpts) -> Vec<ZoomView> {
    let bad = sides.map(|s| masked_pose_bad(s.ev, s.info.mask));
    let names: [Vec<&str>; 2] = sides.map(|s| s.ctx.poses.iter().map(|p| p.name.as_str()).collect());
    let rest = sides.map(|s| posed(s.model, s.ctx, None, s.w));
    rank_rows(&names[0], &bad[0], &names[1], &bad[1], o.rows)
        .into_iter()
        .map(|row| {
            let picks = [row.a, row.b];
            let mesh = [0, 1].map(|k| mesh_at(sides[k], picks[k], 1.0));
            let flags = [0, 1].map(|k| match picks[k] {
                Some((p, _)) => failing(sides[k].model, &sides[k].ev.bad_in_pose[p], sides[k].info.mask),
                None => vec![false; sides[k].model.nverts()],
            });
            let totals = [0, 1].map(|k| flags[k].iter().filter(|f| **f).count());
            let mut best: Option<(f64, View, &'static str, [Fit; 2])> = None;
            for &(yaw_deg, pitch_deg, name) in ZOOM_VIEWS {
                let view = View { yaw_deg, pitch_deg };
                let Some(fits) =
                    zoom_fits(&view, [&rest[0], &rest[1]], [&mesh[0], &mesh[1]], [&flags[0], &flags[1]], o.cell_w, o.cell_h, ZOOM_MIN)
                else {
                    break;
                };
                let mut seen = 0.0;
                for k in 0..2 {
                    if totals[k] > 0 {
                        let half = Fit { ppu: fits[k].ppu * 0.5, ..fits[k] };
                        let n = visible(&mesh[k], &sides[k].model.tris, &flags[k], &view, half, o.cell_w / 2, o.cell_h / 2);
                        seen += n as f64 / totals[k] as f64;
                    }
                }
                if best.as_ref().is_none_or(|b| seen > b.0 + 0.05) {
                    best = Some((seen, view, name, fits));
                }
            }
            let (view, view_name, fits) = match best {
                Some((_, v, n, f)) => (v, n, f),
                None => {
                    let v = View::default();
                    (v, ZOOM_VIEWS[0].2, [0, 1].map(|k| fit(&v, &[&mesh[k]], o.cell_w, o.cell_h)))
                }
            };
            ZoomView { row, view, view_name, fits, flags }
        })
        .collect()
}

/// "POSE: A 86 BAD VERTS, B 12", the larger count in red, and the camera.
fn row_label(img: &mut Image, y: usize, v: &ZoomView, labels: [&str; 2]) {
    let (na, nb) = v.row.counts();
    let picks = [v.row.a, v.row.b];
    let part = |k: usize, tail: &str| match picks[k] {
        Some((_, n)) => format!("{} {n}{tail}", labels[k]),
        None => format!("{} HAS NO SUCH POSE", labels[k]),
    };
    let pieces = [
        (format!("{}: ", v.row.name), FG),
        (part(0, " BAD VERTS"), if na > nb { RED } else { FG }),
        (", ".to_string(), FG),
        (part(1, ""), if nb > na { RED } else { FG }),
    ];
    let len: usize = pieces.iter().map(|p| p.0.chars().count()).sum();
    let scale = if len * 18 + 24 <= img.w { 3 } else { 2 };
    let mut x = 12;
    for (text, col) in &pieces {
        img.text(x, y + 10, text, scale, *col);
        x += text_width(text, scale);
    }
    img.text(12, y + 40, &format!("SEEN FROM {}", v.view_name), 2, DIM);
}

/// One rig's cell: the close-up in clay, the whole figure in the corner
/// with the crop boxed, the rig's name.
#[allow(clippy::too_many_arguments)]
fn zoom_cell(
    img: &mut Image,
    cx: usize,
    cy: usize,
    o: &ZoomOpts,
    s: &Side,
    mats: &[Mat4],
    face: &[([f64; 3], f64)],
    view: &View,
    f: Fit,
    label: &str,
) {
    let tris = &s.model.tris;
    let mesh = &lbs(s.model, mats, s.w)[..];
    img.rect(cx + 2, cy, o.cell_w - 4, o.cell_h - 4, [32, 35, 42]);
    draw_clay(img, cx + 2, cy, o.cell_w - 4, o.cell_h - 4, mesh, tris, face, view, f);
    draw_bones(img, cx + 2, cy, o.cell_w - 4, o.cell_h - 4, &bones_at(s, mats), view, f);
    let (iw, ih) = (o.cell_w / 5, o.cell_h / 3);
    let (ix, iy) = (cx + o.cell_w - iw - 8, cy + 6);
    img.rect(ix, iy, iw, ih, BG);
    let fi = fit(view, &[mesh], iw, ih);
    draw_clay(img, ix, iy, iw, ih, mesh, tris, face, view, fi);
    let (hw, hh) = (o.cell_w as f64 * 0.5 / f.ppu, o.cell_h as f64 * 0.5 / f.ppu);
    let px = |x: f64| ((x - fi.cx) * fi.ppu + iw as f64 * 0.5).clamp(0.0, iw as f64) as usize + ix;
    let py = |y: f64| ((fi.cy - y) * fi.ppu + ih as f64 * 0.5).clamp(0.0, ih as f64) as usize + iy;
    outline(img, px(f.cx - hw), py(f.cy + hh), px(f.cx + hw), py(f.cy - hh), BOX);
    img.text(cx + 12, cy + o.cell_h - 30, label, 2, FG);
}

/// A/B sheet for judging two rigs at a glance: a headline saying which is
/// better, then one pose a row, the pose with the most failing vertices on
/// either rig first. Both rigs are drawn from the same camera, cropped to
/// the same part of the body (where they fail), in grey clay under a side
/// light with the failing faces solid red; a small whole figure in each
/// cell's corner boxes the crop. A last row shows where each rig loses its
/// score, faults no single pose paints included.
pub fn compare_zoom(a: &Side, b: &Side, o: &ZoomOpts) -> Image {
    let sides = [a, b];
    let labels = [o.label_a, o.label_b];
    let notes = [o.note_a, o.note_b];
    let views = zoom_views(sides, o);
    let w = 2 * o.cell_w;
    let mut img = Image::new(w, ZHEADER + (views.len().max(1) + 1) * (ZROW + o.cell_h), BG);

    let totals = sides.map(|s| masked_pose_bad(s.ev, s.info.mask).iter().sum::<usize>());
    let (head, detail) = zoom_headline(&a.info, &b.info, labels[0], labels[1], totals[0], totals[1]);
    let scale = if text_width(&head, 4) + 24 <= w { 4 } else { 3 };
    img.text(12, 12, &fit_text(&head, (w - 24) / (6 * scale)), scale, FG);
    img.text(12, 52, &fit_text(&detail, (w - 24) / 12), 2, FG);
    img.text(12, 78, "ONE POSE A ROW, WORST FIRST. SAME CAMERA AND CROP ON BOTH. RED = FAILS IN THAT POSE.", 2, DIM);
    for (k, side) in sides.iter().enumerate() {
        let x = 12 + k * o.cell_w;
        img.text(x, 106, &fit_text(labels[k], (o.cell_w - 24) / 18), 3, FG);
        let (line, col) = if side.info.pass {
            (format!("{:.1}/100 PASS", side.info.score), GREEN)
        } else {
            (format!("{:.1}/100 FAIL, {} FAILING FINDINGS", side.info.score, side.info.fails), RED)
        };
        let lx = x + text_width(labels[k], 3).min(o.cell_w / 2) + 18;
        img.text(lx, 110, &line, 2, col);
        img.text(x, 138, &fit_text(side.info.title, (o.cell_w - 24) / 12), 2, DIM);
        img.text(x, 160, &fit_text(notes[k], (o.cell_w - 24) / 12), 2, DIM);
    }
    if views.is_empty() {
        img.text(12, ZHEADER + 12, "NO POSE FAILS ON EITHER RIG", 3, GREEN);
    }

    for (r, v) in views.iter().enumerate() {
        let y = ZHEADER + r * (ZROW + o.cell_h);
        row_label(&mut img, y, v, labels);
        let picks = [v.row.a, v.row.b];
        for k in 0..2 {
            let mats = mats_at(sides[k], picks[k], 1.0);
            let face = fault_faces(&sides[k].model.tris, &v.flags[k], 1.0);
            zoom_cell(&mut img, k * o.cell_w, y + ZROW, o, sides[k], &mats, &face, &v.view, v.fits[k], labels[k]);
            if picks[k].is_none() {
                img.text(k * o.cell_w + 14, y + ZROW + 12, "NO SUCH POSE: AT REST", 2, DIM);
            }
        }
    }

    // Where the score goes: every vertex's deformation energy over all
    // poses (speckled and bleeding weights included), whole figure at rest.
    let y = ZHEADER + views.len().max(1) * (ZROW + o.cell_h);
    img.text(12, y + 10, "WHERE EACH RIG LOSES ITS SCORE", 3, FG);
    img.text(12, y + 40, "ALL POSES TOGETHER, ONE SCALE FOR BOTH: GREY IS CLEAN, YELLOW STRAINED, RED BROKEN", 2, DIM);
    let view = View::default();
    for k in 0..2 {
        let (cx, cy) = (k * o.cell_w, y + ZROW);
        let mesh = mesh_at(sides[k], None, 0.0);
        img.rect(cx + 2, cy, o.cell_w - 4, o.cell_h - 4, [32, 35, 42]);
        let face = heat_faces(&sides[k].model.tris, &sides[k].ev.energy);
        let f = fit(&view, &[&mesh], o.cell_w - 4, o.cell_h - 4);
        draw_clay(&mut img, cx + 2, cy, o.cell_w - 4, o.cell_h - 4, &mesh, &sides[k].model.tris, &face, &view, f);
        img.text(cx + 12, cy + o.cell_h - 30, labels[k], 2, FG);
    }
    img
}

/// How far into its pose a clip is at frame `f` (0 rest, 1 the pose): it
/// eases in, holds, eases back and rests, so the clip loops.
pub fn clip_phase(f: usize) -> f64 {
    let ease = |x: f64| x * x * (3.0 - 2.0 * x);
    match f % CLIP_FRAMES {
        f @ 0..24 => ease(f as f64 / 24.0),
        24..34 => 1.0,
        f @ 34..54 => ease(1.0 - (f - 34) as f64 / 20.0),
        _ => 0.0,
    }
}

/// A two-second loop for every row of the zoomed compare: both rigs move
/// from rest into the row's pose and back, on the row's camera and crop,
/// the failing faces reddening as the pose arrives. `sink(row, pose name,
/// frame, image)` gets `CLIP_FRAMES` frames a row, to be played at
/// `CLIP_FPS`.
pub fn compare_zoom_clips(a: &Side, b: &Side, o: &ZoomOpts, mut sink: impl FnMut(usize, &str, usize, &Image)) {
    let sides = [a, b];
    let labels = [o.label_a, o.label_b];
    for (r, v) in zoom_views(sides, o).iter().enumerate() {
        let picks = [v.row.a, v.row.b];
        for f in 0..CLIP_FRAMES {
            let t = clip_phase(f);
            let mut img = Image::new(2 * o.cell_w, ZROW + o.cell_h, BG);
            row_label(&mut img, 0, v, labels);
            for k in 0..2 {
                let mats = mats_at(sides[k], picks[k], t);
                let face = fault_faces(&sides[k].model.tris, &v.flags[k], t);
                zoom_cell(&mut img, k * o.cell_w, ZROW, o, sides[k], &mats, &face, &v.view, v.fits[k], labels[k]);
            }
            sink(r, &v.row.name, f, &img);
        }
    }
}

// ---- Probe: unmarked loops of one rig, for a person to judge ----

/// One case for a person to judge without being told the answer: a pose
/// of one rig, close on where its skin stretches most.
pub struct ProbeCase {
    pub pose: usize,
    pub name: String,
    /// The third-largest vertex stretch in the pose (posed edge length
    /// over rest): the stretch limit at which the pose starts to fail.
    pub stretch: f64,
    /// Vertices the gate fails in this pose as it stands.
    pub bad: usize,
    view: View,
    fit: Fit,
}

/// Per vertex, the largest stretch among its edges in a posed mesh.
fn edge_stretch(model: &Model, ctx: &Ctx, pos: &[Vec3]) -> Vec<f64> {
    let mut out = vec![1.0f64; model.nverts()];
    for (e, r0) in model.edges.iter().zip(&ctx.rest_edge) {
        let (a, b) = (e[0] as usize, e[1] as usize);
        let s = (pos[a] - pos[b]).len() / r0.max(1e-12);
        out[a] = out[a].max(s);
        out[b] = out[b].max(s);
    }
    out
}

/// Cases spread over a range of stretch, one pose for each of `targets`
/// (the pose whose stretch is nearest; a pose and its left or right twin
/// are used once), so a person's "fine" and "broken" can be laid against
/// the number.
pub fn probe_cases(s: &Side, o: &ZoomOpts, targets: &[f64]) -> Vec<ProbeCase> {
    let bad = masked_pose_bad(s.ev, s.info.mask);
    let rest = posed(s.model, s.ctx, None, s.w);
    // (stretch, per-vertex stretch) for every pose.
    let measured: Vec<(f64, Vec<f64>)> = (0..s.ctx.poses.len())
        .map(|p| {
            let st = edge_stretch(s.model, s.ctx, &mesh_at(s, Some((p, 0)), 1.0));
            let mut top = st.clone();
            top.sort_by(|a, b| b.total_cmp(a));
            (top.get(2).copied().unwrap_or(1.0), st)
        })
        .collect();
    let stem = |n: &str| n.trim_end_matches(".L").trim_end_matches(".R").to_string();
    let mut used: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for &t in targets {
        let pick = (0..measured.len())
            .filter(|&p| !used.contains(&stem(&s.ctx.poses[p].name)))
            .min_by(|&a, &b| (measured[a].0 / t).ln().abs().total_cmp(&(measured[b].0 / t).ln().abs()).then(a.cmp(&b)));
        let Some(p) = pick else { break };
        // Too far from the target to stand for it.
        if (measured[p].0 / t).ln().abs() > 0.25 {
            continue;
        }
        used.push(stem(&s.ctx.poses[p].name));
        let (peak, st) = &measured[p];
        let focus: Vec<bool> = st.iter().map(|&x| x >= (0.75 * peak).max(1.3)).collect();
        let mesh = mesh_at(s, Some((p, 0)), 1.0);
        let total = focus.iter().filter(|f| **f).count().max(1);
        let mut best: Option<(f64, View, Fit)> = None;
        for &(yaw_deg, pitch_deg, _) in ZOOM_VIEWS {
            let view = View { yaw_deg, pitch_deg };
            let Some(fits) = zoom_fits(&view, [&rest, &rest], [&mesh, &mesh], [&focus, &focus], o.cell_w, o.cell_h, PROBE_MIN) else {
                break;
            };
            let half = Fit { ppu: fits[0].ppu * 0.5, ..fits[0] };
            let seen = visible(&mesh, &s.model.tris, &focus, &view, half, o.cell_w / 2, o.cell_h / 2) as f64 / total as f64;
            if best.as_ref().is_none_or(|b| seen > b.0 + 0.05) {
                best = Some((seen, view, fits[0]));
            }
        }
        let (view, fit) = match best {
            Some((_, v, f)) => (v, f),
            None => (View::default(), fit(&View::default(), &[&mesh], o.cell_w, o.cell_h)),
        };
        out.push(ProbeCase { pose: p, name: s.ctx.poses[p].name.clone(), stretch: *peak, bad: bad[p], view, fit });
    }
    out
}

/// A case's two-second loop: the rig alone in clay, from rest into the
/// pose and back, with only the case's number on it. Nothing is marked and
/// nothing says what the checker thinks. `sink(frame, image)`.
pub fn probe_clip(s: &Side, c: &ProbeCase, number: usize, o: &ZoomOpts, mut sink: impl FnMut(usize, &Image)) {
    let face = vec![(CLAY, 0.0); s.model.tris.len()];
    for f in 0..CLIP_FRAMES {
        let mut img = Image::new(o.cell_w, ZROW + o.cell_h, BG);
        img.text(12, 12, &format!("CASE {number:02}"), 5, FG);
        let mats = mats_at(s, Some((c.pose, 0)), clip_phase(f));
        zoom_cell(&mut img, 0, ZROW, o, s, &mats, &face, &c.view, c.fit, "");
        sink(f, &img);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_rank_by_the_larger_count() {
        let rows = rank_rows(
            &["arm_up.L", "hip_forward.L", "knee.L", "calm"],
            &[12, 3, 40, 0],
            &["hip_forward.L", "arm_up.L", "calm", "finger_curl.L"],
            &[86, 9, 0, 5],
            3,
        );
        let got: Vec<(&str, (usize, usize))> = rows.iter().map(|r| (r.name.as_str(), r.counts())).collect();
        // By the larger of the two counts, not the sum; the pose neither
        // rig fails is left out, and the fourth-worst is cut.
        assert_eq!(got, vec![("hip_forward.L", (3, 86)), ("knee.L", (40, 0)), ("arm_up.L", (12, 9))]);
        // A pose only one rig has is kept, and says so.
        assert_eq!(rows[1].b, None);
        let all = rank_rows(&["a"], &[1], &["b"], &[2], 9);
        assert_eq!(all.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["b", "a"]);
        assert_eq!((all[0].a, all[0].b), (None, Some((0, 2))));
    }

    #[test]
    fn the_headline_names_the_better_rig() {
        let mask = [];
        let info = |score, pass, fails| SheetInfo { title: "", mask: &mask, score, pass, fails };
        let (h, d) = zoom_headline(&info(15.1, false, 7), &info(59.9, false, 2), "TRIPO", "OURS", 610, 140);
        assert_eq!(h, "OURS IS BETTER: 59.9 VS 15.1 OUT OF 100");
        assert_eq!(d, "OURS VS TRIPO: FAILING FINDINGS 2 VS 7, BAD VERTS OVER ALL POSES 140 VS 610");
        // The gate comes before the score.
        let (h, _) = zoom_headline(&info(70.0, true, 0), &info(90.0, false, 1), "A", "B", 0, 4);
        assert_eq!(h, "A PASSES THE GATE, B DOES NOT");
        let (h, _) = zoom_headline(&info(44.9, false, 2), &info(80.0, false, 3), "TRIPO", "OURS", 30, 41);
        assert_eq!(h, "MIXED: OURS SCORES 80.0 VS 44.9 BUT FAILS MORE");
        let (h, _) = zoom_headline(&info(61.0, false, 2), &info(61.4, false, 2), "A", "B", 9, 9);
        assert!(h.starts_with("ABOUT EVEN"), "{h}");
    }
}

/// One finding of the headline (`owner.rs`) as a sheet a person can read
/// at a glance: the whole body in the frame where it is worst, then a
/// close-up of it there and at rest from the same camera, in the
/// character's own paint with the finding's skin in red.
pub fn finding_sheet(model: &Model, ctx: &Ctx, w: &Weights, f: &crate::owner::Finding, paint: &crate::paint::Paint, title: &str) -> Image {
    let (pw, ph, side) = (460usize, 640usize, 360usize);
    let top = 96usize;
    let (iw, ih) = (side + 2 * pw + 40, top + ph + 44);
    let mut img = Image::new(iw, ih, BG);
    let posed = lbs(model, &ctx.mats[f.worst_pose.min(ctx.mats.len().saturating_sub(1))], w);
    let mut is_f = vec![false; model.nverts()];
    for &v in &f.vertices {
        is_f[v as usize] = true;
    }
    let colors: Vec<[f64; 3]> = (0..model.nverts())
        .map(|v| {
            if is_f[v] {
                return BAD;
            }
            let c = paint.rgb[v];
            let l = 0.3 * c[0] as f64 + 0.59 * c[1] as f64 + 0.11 * c[2] as f64;
            [
                0.15 + 0.6 * (0.7 * c[0] as f64 + 0.3 * l),
                0.15 + 0.6 * (0.7 * c[1] as f64 + 0.3 * l),
                0.15 + 0.6 * (0.7 * c[2] as f64 + 0.3 * l),
            ]
        })
        .collect();
    // The camera that sees most of the finding (a hand inside the body is
    // best seen from where the hand went in).
    let view = best_view(&posed, &model.tris, &is_f);
    // Whole body.
    let full = fit(&view, &[&posed], side, ph);
    draw_mesh(&mut img, 0, top, side, ph, &posed, &model.tris, &colors, &view, full);
    // The crop: the finding's extent, doubled, square-ish to the panel.
    let pts: Vec<Vec3> = f.vertices.iter().map(|&v| view.to_cam(posed[v as usize])).collect();
    let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, 0.0), v3(f64::MIN, f64::MIN, 0.0));
    for q in &pts {
        lo = lo.min(*q);
        hi = hi.max(*q);
    }
    let body_h = crate::owner::height(model);
    let span = ((hi.x - lo.x).max(hi.y - lo.y) * 2.0).max(0.18 * body_h);
    let (cx, cy) = (0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y));
    let ppu = (pw as f64 / span).min(ph as f64 / span);
    let zoom = Fit { cx, cy, ppu };
    draw_mesh(&mut img, side + 20, top, pw, ph, &posed, &model.tris, &colors, &view, zoom);
    // The same crop at rest: rest positions moved so the finding's centre
    // sits where it does posed.
    let rc = f.vertices.iter().fold(Vec3::ZERO, |s, &v| s + model.rest[v as usize]) / f.vertices.len().max(1) as f64;
    let pc = f.vertices.iter().fold(Vec3::ZERO, |s, &v| s + posed[v as usize]) / f.vertices.len().max(1) as f64;
    let rest: Vec<Vec3> = model.rest.iter().map(|p| *p - rc + pc).collect();
    draw_mesh(&mut img, side + pw + 40, top, pw, ph, &rest, &model.tris, &colors, &view, zoom);
    // The crop's box on the whole body.
    let to_px =
        |x: f64, y: f64| (((x - full.cx) * full.ppu + side as f64 * 0.5), ((full.cy - y) * full.ppu + ph as f64 * 0.5 + top as f64));
    let (bx0, by0) = to_px(cx - 0.5 * pw as f64 / ppu, cy + 0.5 * ph as f64 / ppu);
    let (bx1, by1) = to_px(cx + 0.5 * pw as f64 / ppu, cy - 0.5 * ph as f64 / ppu);
    let clampx = |x: f64| x.clamp(0.0, (side - 2) as f64) as usize;
    let clampy = |y: f64| y.clamp(top as f64, (top + ph - 2) as f64) as usize;
    let (x0, x1, y0, y1) = (clampx(bx0), clampx(bx1), clampy(by0), clampy(by1));
    img.rect(x0, y0, x1.saturating_sub(x0).max(2), 2, FG);
    img.rect(x0, y1, x1.saturating_sub(x0).max(2), 2, FG);
    img.rect(x0, y0, 2, y1.saturating_sub(y0).max(2), FG);
    img.rect(x1, y0, 2, y1.saturating_sub(y0).max(2), FG);
    img.text(12, 12, &fit_text(&format!("{}: {}", f.kind.replace('_', " "), f.part), (iw - 24) / 18), 3, RED);
    img.text(12, 44, &fit_text(&f.detail.replace('\u{b2}', "2"), (iw - 24) / 12), 2, FG);
    img.text(
        12,
        70,
        &fit_text(&format!("{:.0} cm2 of skin ({:.2}% of the surface), on a 1.75 m body", f.area_cm2, f.area_pct), (iw - 24) / 12),
        2,
        DIM,
    );
    img.text(side + 28, top + 8, &fit_text(&format!("in {}", f.worst), pw / 12), 2, FG);
    img.text(side + pw + 48, top + 8, "at rest", 2, FG);
    img.text(12, top + ph + 12, &fit_text(title, (iw - 24) / 12), 2, DIM);
    img
}

/// Of 36 cameras round the body (12 headings, 3 heights), the one whose
/// view shows the most pixels of the marked triangles' skin, over a
/// small depth-tested raster.
pub fn best_view(pos: &[Vec3], tris: &[[u32; 3]], marked: &[bool]) -> View {
    let mark: Vec<bool> = tris.iter().map(|t| t.iter().any(|&i| marked[i as usize])).collect();
    let mut best = (0usize, View::default());
    for pitch in [10.0, 40.0, -25.0] {
        for k in 0..12 {
            let view = View { yaw_deg: k as f64 * 30.0, pitch_deg: pitch };
            let cam: Vec<Vec3> = pos.iter().map(|p| view.to_cam(*p)).collect();
            let (mut lo, mut hi) = (v3(f64::MAX, f64::MAX, 0.0), v3(f64::MIN, f64::MIN, 0.0));
            for (t, m) in tris.iter().zip(&mark) {
                if *m {
                    for &i in t {
                        lo = lo.min(cam[i as usize]);
                        hi = hi.max(cam[i as usize]);
                    }
                }
            }
            let n = 96usize;
            let span = (hi.x - lo.x).max(hi.y - lo.y).max(1e-9) * 2.0;
            let (cx, cy) = (0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y));
            let sc = n as f64 / span;
            let mut depth = vec![f64::NEG_INFINITY; n * n];
            let mut owner = vec![false; n * n];
            for (t, m) in tris.iter().zip(&mark) {
                let p = t.map(|i| cam[i as usize]);
                let q = p.map(|c| ((c.x - cx) * sc + n as f64 * 0.5, (cy - c.y) * sc + n as f64 * 0.5, c.z));
                let area = (q[1].0 - q[0].0) * (q[2].1 - q[0].1) - (q[1].1 - q[0].1) * (q[2].0 - q[0].0);
                if area.abs() < 1e-12 {
                    continue;
                }
                let x0 = q.iter().map(|c| c.0).fold(f64::MAX, f64::min).floor().max(0.0) as usize;
                let x1 = (q.iter().map(|c| c.0).fold(f64::MIN, f64::max).ceil() as isize).min(n as isize - 1);
                let y0 = q.iter().map(|c| c.1).fold(f64::MAX, f64::min).floor().max(0.0) as usize;
                let y1 = (q.iter().map(|c| c.1).fold(f64::MIN, f64::max).ceil() as isize).min(n as isize - 1);
                if x1 < 0 || y1 < 0 {
                    continue;
                }
                for py in y0..=y1 as usize {
                    for px in x0..=x1 as usize {
                        let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
                        let w0 = ((q[1].0 - x) * (q[2].1 - y) - (q[1].1 - y) * (q[2].0 - x)) / area;
                        let w1 = ((q[2].0 - x) * (q[0].1 - y) - (q[2].1 - y) * (q[0].0 - x)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let z = w0 * q[0].2 + w1 * q[1].2 + w2 * q[2].2;
                        let i = py * n + px;
                        if z > depth[i] {
                            depth[i] = z;
                            owner[i] = *m;
                        }
                    }
                }
            }
            let seen = owner.iter().filter(|&&o| o).count();
            if seen > best.0 {
                best = (seen, view);
            }
        }
    }
    best.1
}
