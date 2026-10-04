//! P1: pose sheet and A/B compare are byte-stable golden images.
//! Regenerate after an intended render change with
//! `UPDATE_GOLDENS=1 cargo test --release`, then look at the PNGs.

use std::path::PathBuf;
use weightforge_core::fixtures;
use weightforge_core::glb::Glb;
use weightforge_core::metrics::{Ctx, CtxOpts, Eval, evaluate};
use weightforge_core::render::{SheetInfo, SheetOpts, Side, compare, sheet};
use weightforge_core::report::{Report, build, finding_mask};
use weightforge_core::scene::Model;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/golden").join(name)
}

fn load(fault: &str) -> (Model, Ctx, Eval, Report) {
    let bytes = fixtures::to_glb(&fixtures::build(fault)).to_bytes();
    let m = Model::from_glb(Glb::parse(&bytes).unwrap()).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let ev = evaluate(&m, &ctx, &m.weights);
    let rep = build(&format!("mannequin_{fault}.glb"), &m, &ctx, &ev);
    (m, ctx, ev, rep)
}

fn check_golden(name: &str, png: &[u8]) {
    let path = golden(name);
    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, png).unwrap();
        return;
    }
    let want = std::fs::read(&path).unwrap_or_else(|_| panic!("missing golden {}: run with UPDATE_GOLDENS=1", path.display()));
    if want != png {
        let got = path.with_extension("got.png");
        std::fs::write(&got, png).unwrap();
        panic!("{name} differs from golden; wrote {}", got.display());
    }
}

fn fails(r: &Report) -> usize {
    r.findings.iter().filter(|f| f.severity == "fail").count()
}

#[test]
fn sheet_bleed_golden() {
    let (m, ctx, ev, rep) = load("bleed");
    let mask = finding_mask(&ctx, &ev, &rep);
    let info = SheetInfo { title: &rep.file, mask: &mask, score: rep.score, pass: rep.pass, fails: fails(&rep) };
    let img = sheet(&m, &ctx, &ev, &m.weights, &info, &SheetOpts::default());
    check_golden("sheet_bleed.png", &img.png_bytes());
}

#[test]
fn compare_cape_golden() {
    let (ma, ca, ea, ra) = load("clean");
    let (mb, cb, eb, rb) = load("cape_wrong_bone");
    let (ka, kb) = (finding_mask(&ca, &ea, &ra), finding_mask(&cb, &eb, &rb));
    let sa = Side {
        model: &ma,
        ctx: &ca,
        ev: &ea,
        w: &ma.weights,
        info: SheetInfo { title: &ra.file, mask: &ka, score: ra.score, pass: ra.pass, fails: fails(&ra) },
    };
    let sb = Side {
        model: &mb,
        ctx: &cb,
        ev: &eb,
        w: &mb.weights,
        info: SheetInfo { title: &rb.file, mask: &kb, score: rb.score, pass: rb.pass, fails: fails(&rb) },
    };
    let img = compare(&sa, &sb, &SheetOpts::default());
    check_golden("compare_clean_vs_cape.png", &img.png_bytes());
}
