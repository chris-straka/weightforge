//! The fix keeps results by one order everywhere: passing the gate first,
//! then score. Its last guard used to revert to the input on score alone,
//! so a result that passed at a lower score than a failing input (the pick
//! kept it for passing) was thrown away at the end.

use weightforge_core::fix::ranks_below;
use weightforge_core::fixtures;
use weightforge_core::metrics::{Ctx, CtxOpts, evaluate};
use weightforge_core::report::build;
use weightforge_core::scene::Model;

#[test]
fn passing_outranks_failing_at_any_score() {
    let m = Model::from_glb(fixtures::to_glb(&fixtures::build("clean"))).unwrap();
    let ctx = Ctx::new(&m, &CtxOpts::default());
    let rep = build("t", &m, &ctx, &evaluate(&m, &ctx, &m.weights));
    let with = |pass: bool, score: f64| {
        let mut r = rep.clone();
        r.pass = pass;
        r.score = score;
        r
    };
    let (input, fixed) = (with(false, 80.0), with(true, 70.0));
    // The guard reverts when the fix ranks below the input: not here.
    assert!(!ranks_below(&fixed, &input));
    assert!(ranks_below(&input, &fixed));
    // Same verdict: the score decides, ties keep the result.
    assert!(ranks_below(&with(true, 69.9), &fixed));
    assert!(!ranks_below(&with(true, 70.0), &fixed));
    assert!(ranks_below(&with(false, 79.0), &input));
}
