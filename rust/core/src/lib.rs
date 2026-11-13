//! weightforge core: find and fix bad skin weights on rigged GLB characters.
//!
//! Pipeline: [`scene::Model`] loads a GLB (skeleton + welded skinned mesh),
//! [`metrics::Ctx`] poses it through a range-of-motion set and measures what
//! breaks, [`report`] turns that into per-region scores, [`render`] draws pose
//! sheets, and [`fix`] builds weight candidates, scores them on the same poses,
//! and keeps the best per region.

pub mod anatomy;
pub mod fix;
pub mod fixtures;
pub mod font;
pub mod glb;
pub mod helpers;
pub mod math;
pub mod metrics;
pub mod motion;
pub mod owner;
pub mod paint;
pub mod pick;
pub mod poses;
pub mod render;
pub mod report;
pub mod scene;
pub mod skin;
pub mod solve;
pub mod transfer;
pub mod unfuse;
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

/// Where the game's motion comes from for a check.
#[derive(Clone, Debug, Default)]
pub struct MotionSpec {
    /// A rigged GLB whose clips are put on the checked rig by role (the
    /// hero's file). None: the checked file's own clips, if it has any.
    pub source: Option<std::path::PathBuf>,
    /// Clips to play (default `motion::GAME_CLIPS`; with no source, all
    /// of the file's own).
    pub clips: Option<Vec<String>>,
}

/// A whole check: the range-of-motion poses (`ctx`, `ev`, `rep`, whose
/// `headline` holds both sections) and the game's clips beside them.
pub struct Checked {
    pub model: scene::Model,
    pub ctx: metrics::Ctx,
    pub ev: metrics::Eval,
    pub rep: report::Report,
    pub motion: Option<(metrics::Ctx, metrics::Eval, report::Report)>,
}

/// `check` with the headline: the poses without the file's clips, then the
/// clips (the file's own, or `spec.source`'s put on this rig) as a second
/// group, and the owner-eye findings of each (`owner.rs`).
pub fn check_full(path: &Path, opts: &metrics::CtxOpts, spec: &MotionSpec) -> glb::Result<Checked> {
    let model = scene::Model::load(path)?;
    let rom = metrics::CtxOpts { clips: false, poses: opts.poses, th: opts.th.clone(), ..*opts };
    let ctx = metrics::Ctx::new(&model, &rom);
    let ev = metrics::evaluate(&model, &ctx, &model.weights);
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut rep = report::build(&name, &model, &ctx, &ev);
    let mut notes = Vec::new();
    let mv = match &spec.source {
        Some(src) => {
            let sm = scene::Model::load(src)?;
            let sname = src.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let names: Vec<String> = spec.clips.clone().unwrap_or_else(|| motion::GAME_CLIPS.iter().map(|s| s.to_string()).collect());
            match motion::transfer(&sm, &sname, &model, &names, motion::FPS) {
                Ok(m) => Some(m),
                Err(e) => {
                    notes.push(format!("clips from {sname} not played: {e}"));
                    None
                }
            }
        }
        None if opts.clips => Some(motion::own(&model, motion::FPS, spec.clips.as_deref())).filter(|m| !m.poses.is_empty()),
        None => None,
    };
    let mut motion_out = None;
    if let Some(m) = mv {
        notes.extend(m.skipped.iter().map(|s| format!("clip not played: {s}")));
        let cm = ctx.with_poses(&model, m.poses);
        let em = metrics::evaluate_with(&model, &cm, &model.weights, false);
        let rm = report::build(&name, &model, &cm, &em);
        let sec = owner::assess(&model, &cm, &em, &rm, &model.weights, true, "game clips", m.clips, m.source);
        rep.headline.push(sec);
        motion_out = Some((cm, em, rm));
    }
    let sec = owner::assess(&model, &ctx, &ev, &rep, &model.weights, false, "poses", Vec::new(), "range of motion".into());
    rep.headline.push(sec);
    rep.anatomy.extend(notes);
    Ok(Checked { model, ctx, ev, rep, motion: motion_out })
}
