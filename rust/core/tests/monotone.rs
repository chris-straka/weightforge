//! End to end: a better external candidate does not give a worse fix. The
//! candidate slides from the faulted input (t = 1) to the ground truth
//! (t = 0). `pick.rs` proves the exact statement for the region pick
//! (better measured per-region numbers never pick a worse plan); on the
//! mesh, "closer to ground truth" is not quite "better in every region as
//! blended in" (the ground truth can blend into a faulted neighbor less
//! well than a 20% mix), so this allows 0.5 points of slack against the
//! best score so far. Before the exact pick, the greedy lost 7 points this
//! way on a real rig (a better SkinTokens skin: 52 -> 46).

use weightforge_core::fix::{FixOpts, fix};
use weightforge_core::fixtures;
use weightforge_core::metrics::{Ctx, CtxOpts};
use weightforge_core::scene::{Model, VW, Weights, normalize_vw};

fn mix(a: &Weights, b: &Weights, t: f64) -> Weights {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let mut acc: VW = x.iter().map(|&(j, w)| (j, w * t)).collect();
            for &(j, w) in y {
                match acc.iter_mut().find(|e| e.0 == j) {
                    Some(e) => e.1 += w * (1.0 - t),
                    None => acc.push((j, w * (1.0 - t))),
                }
            }
            acc.sort_by(|p, q| q.1.partial_cmp(&p.1).unwrap().then(p.0.cmp(&q.0)));
            acc.truncate(4);
            normalize_vw(acc)
        })
        .collect()
}

#[test]
fn better_external_candidate_does_not_score_worse() {
    for fault in ["elbow_collapse", "knee_wide_falloff", "noise"] {
        let f = fixtures::build(fault);
        let m = Model::from_glb(fixtures::to_glb(&f)).unwrap();
        let ctx = Ctx::new(&m, &CtxOpts::default());
        let (gt_body, gt_cape) = fixtures::ground_truth();
        let mut best = f64::MIN;
        for t in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] {
            let mut c = fixtures::build(fault);
            c.body_w = mix(&f.body_w, &gt_body, t);
            c.cape_w = mix(&f.cape_w, &gt_cape, t);
            let cand = Model::from_glb(fixtures::to_glb(&c)).unwrap();
            let o = FixOpts { external: vec![("cand".into(), &cand)], ..FixOpts::default() };
            let r = fix(fault, &m, &ctx, &o);
            eprintln!("{fault} t {t}: {:.1} -> {:.1}", r.fix.before.score, r.fix.after.score);
            assert!(r.fix.after.score >= r.fix.before.score);
            assert!(r.fix.after.score >= best - 0.5, "{fault}: better candidate (t {t}) scored {} after {best}", r.fix.after.score);
            best = best.max(r.fix.after.score);
        }
    }
}

/// Guaranteed by construction (`fix` also runs without external candidates
/// and keeps the better): any external candidate, even a bad one, never
/// leaves the fix worse than no candidate at all.
#[test]
fn an_external_candidate_never_hurts() {
    let f = fixtures::build("elbow_collapse");
    let m = Model::from_glb(fixtures::to_glb(&f)).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let alone = fix("elbow_collapse", &m, &ctx, &FixOpts::default()).fix.after.score;
    for bad in ["bleed", "cape_wrong_bone", "noise"] {
        let cand = Model::from_glb(fixtures::to_glb(&fixtures::build(bad))).unwrap();
        let o = FixOpts { external: vec![(bad.into(), &cand)], ..FixOpts::default() };
        let r = fix("elbow_collapse", &m, &ctx, &o);
        eprintln!("elbow_collapse + {bad} candidate: {:.1} (alone {alone:.1})", r.fix.after.score);
        assert!(r.fix.after.score >= alone);
    }
}
