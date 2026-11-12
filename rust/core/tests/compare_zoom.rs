//! The zoomed compare (`weights compare --zoom`): a fault has to be big
//! enough to see, on the same crop for both rigs.

use weightforge_core::fixtures;
use weightforge_core::glb::Glb;
use weightforge_core::metrics::{Ctx, CtxOpts, Eval, evaluate};
use weightforge_core::render::{
    CLIP_FRAMES, Image, SheetInfo, SheetOpts, Side, ZoomOpts, clip_phase, compare, compare_zoom, compare_zoom_clips,
};
use weightforge_core::report::{Report, build, finding_mask};
use weightforge_core::scene::Model;
use weightforge_core::skin::{joint_matrices, lbs, pose_at};

fn load(fault: &str) -> (Model, Ctx, Eval, Report) {
    let bytes = fixtures::to_glb(&fixtures::build(fault)).to_bytes();
    let m = Model::from_glb(Glb::parse(&bytes).unwrap()).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let ev = evaluate(&m, &ctx, &m.weights);
    let rep = build(&format!("mannequin_{fault}.glb"), &m, &ctx, &ev);
    (m, ctx, ev, rep)
}

fn fails(r: &Report) -> usize {
    r.findings.iter().filter(|f| f.severity == "fail").count()
}

fn side<'a>(m: &'a Model, c: &'a Ctx, e: &'a Eval, r: &'a Report, k: &'a [u8]) -> Side<'a> {
    Side {
        model: m,
        ctx: c,
        ev: e,
        w: &m.weights,
        info: SheetInfo { title: &r.file, mask: k, score: r.score, pass: r.pass, fails: fails(r) },
    }
}

const HEADER: usize = 186;
const LABEL: usize = 62;

/// Strongly red pixels in the columns x0..x1 of rows `rows` of a zoomed
/// sheet's cells (a clip frame is one row with no header: `top` 0). Row
/// labels, whose counts are red text, are skipped.
fn red(img: &Image, top: usize, cell: usize, rows: std::ops::Range<usize>, x0: usize, x1: usize) -> usize {
    let mut n = 0;
    for r in rows {
        let y0 = top + r * (LABEL + cell) + LABEL;
        for y in y0..(y0 + cell).min(img.h) {
            for x in x0..x1 {
                let o = (y * img.w + x) * 3;
                let (r, g, b) = (img.rgb[o] as i32, img.rgb[o + 1] as i32, img.rgb[o + 2] as i32);
                if r > 150 && g < 90 && b < 90 {
                    n += 1;
                }
            }
        }
    }
    n
}

fn pair() -> [(Model, Ctx, Eval, Report); 2] {
    [load("elbow_collapse"), load("clean")]
}

#[test]
fn a_fault_fills_the_zoomed_sheet_and_a_clean_rig_shows_none() {
    let [(ma, ca, ea, ra), (mb, cb, eb, rb)] = pair();
    let (ka, kb) = (finding_mask(&ca, &ea, &ra), finding_mask(&cb, &eb, &rb));
    let (sa, sb) = (side(&ma, &ca, &ea, &ra, &ka), side(&mb, &cb, &eb, &rb, &kb));
    let o = ZoomOpts { rows: 4, label_a: "COLLAPSED", label_b: "CLEAN", ..Default::default() };
    let zoom = compare_zoom(&sa, &sb, &o);
    assert_eq!(zoom.w, 2 * o.cell_w);
    // The pose rows, then the row that shows where the score goes.
    let rows = (zoom.h - HEADER) / (LABEL + o.cell_h) - 1;
    assert!((1..=4).contains(&rows), "{rows}");

    // The fault is a solid patch: in the worst row the faulty rig's red
    // covers a real share of its cell, not a speck, and every row shows some.
    let cell = (o.cell_w * o.cell_h) as f64;
    let share = |r: usize| red(&zoom, HEADER, o.cell_h, r..r + 1, 0, zoom.w / 2) as f64 / cell;
    assert!(share(0) > 0.01, "worst row: red fills {:.2}% of the cell", share(0) * 100.0);
    assert!((0..rows).all(|r| share(r) > 0.0));
    // The whole-body sheet spends far less of itself on the same fault.
    let whole = compare(&sa, &sb, &SheetOpts::default());
    let whole_red = {
        let mut n = 0;
        for y in 136..whole.h {
            for x in 0..whole.w / 2 {
                let o = (y * whole.w + x) * 3;
                if whole.rgb[o] > 150 && whole.rgb[o + 1] < 90 && whole.rgb[o + 2] < 90 {
                    n += 1;
                }
            }
        }
        n as f64 / ((whole.h - 136) * whole.w / 2) as f64
    };
    let zoom_red = red(&zoom, HEADER, o.cell_h, 0..rows, 0, zoom.w / 2) as f64 / (rows as f64 * cell);
    assert!(zoom_red > 4.0 * whole_red, "red share: whole-body {whole_red:.4}, zoomed {zoom_red:.4}");

    // The clean rig, on the same crops, shows none.
    assert_eq!(red(&zoom, HEADER, o.cell_h, 0..rows, zoom.w / 2, zoom.w), 0);
}

#[test]
fn two_clean_rigs_make_a_sheet_that_says_so() {
    let (m, c, e, r) = load("clean");
    let k = finding_mask(&c, &e, &r);
    let s = || Side {
        model: &m,
        ctx: &c,
        ev: &e,
        w: &m.weights,
        info: SheetInfo { title: &r.file, mask: &k, score: r.score, pass: r.pass, fails: 0 },
    };
    let o = ZoomOpts::default();
    let img = compare_zoom(&s(), &s(), &o);
    // No pose row; the line that says so, and the score row.
    assert_eq!(img.h, HEADER + 2 * (LABEL + o.cell_h));
}

#[test]
fn a_rows_clip_goes_into_the_pose_and_comes_back() {
    let [(ma, ca, ea, ra), (mb, cb, eb, rb)] = pair();
    let (ka, kb) = (finding_mask(&ca, &ea, &ra), finding_mask(&cb, &eb, &rb));
    let (sa, sb) = (side(&ma, &ca, &ea, &ra, &ka), side(&mb, &cb, &eb, &rb, &kb));
    let o = ZoomOpts { rows: 1, ..Default::default() };
    let mut frames: Vec<Image> = Vec::new();
    compare_zoom_clips(&sa, &sb, &o, |row, _, f, img| {
        assert_eq!((row, f), (0, frames.len()));
        frames.push(Image { w: img.w, h: img.h, rgb: img.rgb.clone() });
    });
    assert_eq!(frames.len(), CLIP_FRAMES);
    // Even sides, for the video encoder.
    assert_eq!((frames[0].w % 2, frames[0].h % 2), (0, 0));
    let reds: Vec<usize> = frames.iter().map(|f| red(f, 0, o.cell_h, 0..1, 0, f.w / 2)).collect();
    // At rest nothing is marked; the patch grows into the held pose; the
    // last frame is the first again, so the clip loops.
    assert_eq!(reds[0], 0);
    assert!(reds[28] > 500, "{}", reds[28]);
    assert!(reds[12] < reds[28]);
    assert_eq!(frames[0].rgb, frames[CLIP_FRAMES - 1].rgb);
    // The pose moved the mesh: the held frame is not the rest frame.
    assert_ne!(frames[0].rgb, frames[28].rgb);
    assert_eq!((clip_phase(0), clip_phase(28), clip_phase(CLIP_FRAMES - 1)), (0.0, 1.0, 0.0));
}

#[test]
fn a_pose_taken_all_the_way_is_the_pose() {
    let (m, c, _, _) = load("clean");
    for (p, pose) in c.poses.iter().enumerate().take(6) {
        let full = joint_matrices(&m, &pose_at(&m, pose, 1.0));
        let rest = lbs(&m, &joint_matrices(&m, &pose_at(&m, pose, 0.0)), &m.weights);
        for (a, b) in full.iter().zip(&c.mats[p]) {
            for (x, y) in a.to_cols_vec().iter().zip(b.to_cols_vec()) {
                assert!((x - y).abs() < 1e-9, "{}", pose.name);
            }
        }
        for (a, b) in rest.iter().zip(&m.rest) {
            assert!((*a - *b).len() < 1e-9, "{}", pose.name);
        }
    }
}
