//! Repair bands are metric (feature sizes, not edge rings), so the fix
//! works the same at any mesh density. The mannequin is built at five
//! densities (0.8x to 2x the edges per metre). Its clean twin is the
//! reference at each density: the check's own verdict on ground-truth
//! weights varies a little with tessellation (its armpit sits at the 2.5x
//! stretch limit), and that is the checker, not the fix.

use weightforge_core::fix::{FixOpts, fix};
use weightforge_core::fixtures;
use weightforge_core::metrics::{Ctx, CtxOpts};
use weightforge_core::report::Report;
use weightforge_core::scene::Model;

const DENSITIES: [f64; 5] = [0.8, 1.0, 1.3, 1.6, 2.0];

fn fixed(fault: &str, density: f64) -> (f64, f64, Report) {
    let m = Model::from_glb(fixtures::to_glb(&fixtures::build_at(fault, density))).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let r = fix(fault, &m, &ctx, &FixOpts::default());
    (r.fix.before.score, r.fix.after.score, r.report_after)
}

fn fails(r: &Report) -> Vec<(String, String)> {
    r.findings.iter().filter(|f| f.severity == "fail").map(|f| (f.region.clone(), f.code.clone())).collect()
}

/// Faults with metric inputs (a hard step, a collapse band, a too-wide
/// knee): the fix lands on the clean twin's score at every density. With
/// ring-count bands it fell short on the densest mesh (elbow 86.5 and knee
/// 89.2 vs 90.4 at 2x).
#[test]
fn fix_recovers_the_clean_twin_at_every_density() {
    for &d in &DENSITIES {
        let clean = fixed("clean", d).1;
        for fault in ["no_elbow_falloff", "elbow_collapse", "knee_wide_falloff"] {
            let (before, after, _) = fixed(fault, d);
            eprintln!("x{d} {fault}: {before:.1} -> {after:.1} (clean twin {clean:.1})");
            assert!(after >= clean - 0.5, "x{d} {fault}: fixed {after:.1}, clean twin {clean:.1}");
        }
    }
}

/// The ML-rigger case: every joint blend one edge ring wide, so the input
/// gets sharper (and scores worse) as the mesh gets denser. After the fix
/// no density keeps a failing finding the clean twin does not have.
#[test]
fn ring_wide_blends_are_repaired_at_every_density() {
    for &d in &DENSITIES {
        let clean = fails(&fixed("clean", d).2);
        let (before, after, rep) = fixed("ring_bands", d);
        let left: Vec<_> = fails(&rep).into_iter().filter(|f| !clean.contains(f)).collect();
        eprintln!("x{d} ring_bands: {before:.1} -> {after:.1}, left {left:?}");
        assert!(after > before);
        assert!(left.is_empty(), "x{d}: ring_bands still fails {left:?}");
    }
}
