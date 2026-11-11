//! The zoomed compare (`weights compare --zoom`): a fault has to be big
//! enough to see, on the same crop for both rigs.

use weightforge_core::fixtures;
use weightforge_core::glb::Glb;
use weightforge_core::metrics::{Ctx, CtxOpts, Eval, evaluate};
use weightforge_core::render::{Image, SheetInfo, SheetOpts, Side, ZoomOpts, compare, compare_zoom};
use weightforge_core::report::{Report, build, finding_mask};
use weightforge_core::scene::Model;

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

/// Strongly red pixels in the columns x0..x1 of the sheet's cells: the
/// header and the row labels (red text) are skipped. `top` is the header's
/// height, `label` each row's label band, `cell` the cell height.
fn red(img: &Image, (top, label, cell): (usize, usize, usize), x0: usize, x1: usize) -> usize {
    let mut n = 0;
    for y in top..img.h {
        if (y - top) % (label + cell) < label {
            continue;
        }
        for x in x0..x1 {
            let o = (y * img.w + x) * 3;
            let (r, g, b) = (img.rgb[o] as i32, img.rgb[o + 1] as i32, img.rgb[o + 2] as i32);
            if r > 150 && g < 90 && b < 90 {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn a_fault_fills_far_more_of_the_zoomed_sheet() {
    let (ma, ca, ea, ra) = load("elbow_collapse");
    let (mb, cb, eb, rb) = load("clean");
    let (ka, kb) = (finding_mask(&ca, &ea, &ra), finding_mask(&cb, &eb, &rb));
    let (sa, sb) = (side(&ma, &ca, &ea, &ra, &ka), side(&mb, &cb, &eb, &rb, &kb));
    let whole = compare(&sa, &sb, &SheetOpts::default());
    let o = ZoomOpts { rows: 4, label_a: "COLLAPSED", label_b: "CLEAN", ..Default::default() };
    let zoom = compare_zoom(&sa, &sb, &o);
    assert_eq!(zoom.w, 2 * o.cell_w);

    // Per row shown, the faulty rig's red covers several times the area it
    // does on the whole-body sheet (which also spends rows on mild poses).
    // (header, label band, cell height) of each sheet's layout.
    let whole_grid = (136, 30, SheetOpts::default().cell_h - 30);
    let zoom_grid = (164, 62, o.cell_h);
    let rows = |img: &Image, g: (usize, usize, usize)| (img.h - g.0) / (g.1 + g.2);
    let per_row = |img: &Image, g| red(img, g, 0, img.w / 2) as f64 / rows(img, g) as f64;
    assert!((1..=4).contains(&rows(&zoom, zoom_grid)));
    let (before, after) = (per_row(&whole, whole_grid), per_row(&zoom, zoom_grid));
    assert!(after > 4.0 * before, "red per row: whole-body {before:.0}, zoomed {after:.0}");

    // The clean rig, on the same crops, shows none.
    assert_eq!(red(&zoom, zoom_grid, zoom.w / 2, zoom.w), 0);
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
    let img = compare_zoom(&s(), &s(), &ZoomOpts::default());
    assert_eq!(red(&img, (164, 62, 420), 0, img.w), 0);
}
