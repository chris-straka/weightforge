//! weightforge core: find and fix bad skin weights on rigged GLB characters.
//!
//! Pipeline: [`scene::Model`] loads a GLB (skeleton + welded skinned mesh),
//! [`metrics::Ctx`] poses it through a range-of-motion set and measures what
//! breaks, [`report`] turns that into per-region scores, [`render`] draws pose
//! sheets, and [`fix`] builds weight candidates, scores them on the same poses,
//! and keeps the best per region.

pub mod fix;
pub mod fixtures;
pub mod font;
pub mod glb;
pub mod helpers;
pub mod math;
pub mod metrics;
pub mod poses;
pub mod render;
pub mod report;
pub mod scene;
pub mod skin;
pub mod solve;
pub mod transfer;
pub mod voxel;

use std::path::Path;

/// Loads, poses, and scores a GLB. Returns the model, context, evaluation,
/// and report (callers that render or fix reuse the first three).
pub fn check(path: &Path, opts: &metrics::CtxOpts) -> glb::Result<(scene::Model, metrics::Ctx, metrics::Eval, report::Report)> {
    let model = scene::Model::load(path)?;
    let ctx = metrics::Ctx::new(&model, opts);
    let ev = metrics::evaluate(&model, &ctx, &model.weights);
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let rep = report::build(&name, &model, &ctx, &ev);
    Ok((model, ctx, ev, rep))
}
