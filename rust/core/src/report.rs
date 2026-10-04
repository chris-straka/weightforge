//! Check report: per-region scores, findings, worst poses. JSON for tools,
//! a short text summary for people.

use crate::metrics::{
    Ctx, Eval, F_BLEED, F_FOLLOW, F_INTERSECT, F_NOISE, F_PIECE, F_STRETCH, F_THIN, F_UNWEIGHTED, FLAG_NAMES, Thresholds,
};
use crate::scene::Model;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct RegionReport {
    pub name: String,
    pub label: String,
    pub verts: usize,
    pub bad: usize,
    pub score: f64,
    pub energy: f64,
    pub flags: BTreeMap<String, usize>,
    pub worst_pose: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub code: String,
    pub severity: String,
    pub region: String,
    pub verts: usize,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PoseReport {
    pub pose: String,
    pub bad: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub tool: String,
    pub version: String,
    pub file: String,
    pub class: String,
    pub verts: usize,
    pub tris: usize,
    pub bones: usize,
    pub poses: usize,
    pub scale: f64,
    pub score: f64,
    pub pass: bool,
    pub findings: Vec<Finding>,
    pub regions: Vec<RegionReport>,
    pub worst_poses: Vec<PoseReport>,
    pub thresholds: Thresholds,
}

pub fn energy_to_score(e: f64) -> f64 {
    (1000.0 / (1.0 + 10.0 * e)).round() / 10.0
}

fn code_for(flag: u8) -> (&'static str, &'static str) {
    match flag {
        F_STRETCH => ("D_STRETCH", "fail"),
        F_THIN => ("D_VOLUME", "fail"),
        F_BLEED => ("D_BLEED", "fail"),
        F_NOISE => ("D_NOISE", "fail"),
        F_INTERSECT => ("D_INTERSECT", "warn"),
        F_UNWEIGHTED => ("D_UNWEIGHTED", "fail"),
        F_FOLLOW => ("D_PIECE_FOLLOW", "fail"),
        _ => ("D_PIECE_BONES", "fail"),
    }
}

pub fn region_energy(ctx: &Ctx, ev: &Eval) -> Vec<f64> {
    region_energy_of(ctx, &ev.energy)
}

/// Mean per-vertex energy of each region.
pub fn region_energy_of(ctx: &Ctx, energy: &[f64]) -> Vec<f64> {
    let mut sum = vec![0.0; ctx.regions.len()];
    let mut cnt = vec![0usize; ctx.regions.len()];
    for (v, &r) in ctx.vregion.iter().enumerate() {
        sum[r as usize] += energy[v];
        cnt[r as usize] += 1;
    }
    sum.iter().zip(&cnt).map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect()
}

pub fn build(file: &str, model: &Model, ctx: &Ctx, ev: &Eval) -> Report {
    let nr = ctx.regions.len();
    let mut verts = vec![0usize; nr];
    let mut bad = vec![0usize; nr];
    let mut flagc: Vec<BTreeMap<u8, usize>> = vec![BTreeMap::new(); nr];
    let mut pose_votes: Vec<BTreeMap<u32, usize>> = vec![BTreeMap::new(); nr];
    // (region, flag) -> pose -> count of verts with that flag in that pose.
    let mut flag_votes: BTreeMap<(usize, u8), BTreeMap<u32, usize>> = BTreeMap::new();
    for (p, list) in ev.bad_in_pose.iter().enumerate() {
        for &(v, f) in list {
            let r = ctx.vregion[v as usize] as usize;
            for bit in [F_STRETCH, F_THIN, F_INTERSECT, F_FOLLOW] {
                if f & bit != 0 {
                    *flag_votes.entry((r, bit)).or_default().entry(p as u32).or_default() += 1;
                }
            }
        }
    }
    let mut bleed_to: Vec<BTreeMap<u32, (usize, f64)>> = vec![BTreeMap::new(); nr];
    for v in 0..model.nverts() {
        let r = ctx.vregion[v] as usize;
        verts[r] += 1;
        let f = ev.flags[v];
        if f != 0 {
            bad[r] += 1;
            for (bit, _) in FLAG_NAMES {
                if f & bit != 0 {
                    *flagc[r].entry(bit).or_default() += 1;
                }
            }
            if f & (F_STRETCH | F_THIN | F_INTERSECT | F_FOLLOW) != 0 {
                *pose_votes[r].entry(ev.worst_pose[v]).or_default() += 1;
            }
        }
        if let Some((j, _, ex)) = ev.bleed[v] {
            let e = bleed_to[r].entry(j).or_default();
            e.0 += 1;
            e.1 = e.1.max(ex);
        }
    }
    let mut attached = vec![0usize; nr];
    for m in ctx.piece_matches.iter().filter(|m| m.3) {
        attached[ctx.vregion[m.0 as usize] as usize] += 1;
    }
    let energies = region_energy(ctx, ev);
    let th = &ctx.th;
    let mut findings = Vec::new();
    let mut regions = Vec::new();
    for r in 0..nr {
        let reg = &ctx.regions[r];
        for (&bit, &n) in &flagc[r] {
            let min = if bit == F_UNWEIGHTED || bit == F_PIECE { 1 } else { th.region_min_bad };
            if n < min {
                continue;
            }
            let (code, mut sev) = code_for(bit);
            // A piece that strays in only a few spots is an artistic choice
            // (say, a cape left off the legs); most of it straying is a bug.
            if bit == F_FOLLOW && n * 2 < attached[r] {
                sev = "warn";
            }
            let detail = match bit {
                F_BLEED => {
                    let (j, (c, ex)) = bleed_to[r]
                        .iter()
                        .max_by_key(|(j, (c, _))| (*c, usize::MAX - **j as usize))
                        .map(|(j, x)| (*j, *x))
                        .unwrap_or((0, (0, 0.0)));
                    format!(
                        "{c} verts weighted to {} ({:.0}% of body size farther than the nearest bone)",
                        model.skel.names[j as usize],
                        100.0 * ex / model.scale
                    )
                }
                F_STRETCH | F_THIN | F_INTERSECT | F_FOLLOW => {
                    let p = flag_votes
                        .get(&(r, bit))
                        .and_then(|m| m.iter().max_by_key(|(p, c)| (**c, u32::MAX - **p)))
                        .map(|(p, _)| ctx.poses[*p as usize].name.clone())
                        .unwrap_or_default();
                    match bit {
                        F_STRETCH => format!("edges stretch past {:.1}x, worst in {p}", th.stretch),
                        F_FOLLOW => format!(
                            "{n} of {} attached verts drift > {:.0}% of body size from the body surface they hang on, worst in {p}",
                            attached[r],
                            th.follow * 100.0
                        ),
                        F_THIN => format!(
                            "collapses below {:.0}% thickness over > {:.1} limb radii, worst in {p}",
                            th.thin * 100.0,
                            th.thin_extent
                        ),
                        _ => format!("passes through another body part, worst in {p}"),
                    }
                }
                F_NOISE => "jagged / speckled weights".to_string(),
                F_UNWEIGHTED => "vertices with no weights".to_string(),
                _ => {
                    let names = ev.piece_bones.iter().find(|(m, _)| reg.label == *m).map(|(_, b)| b.join(", ")).unwrap_or_default();
                    format!("piece skinned to non-body bones: {names}")
                }
            };
            findings.push(Finding { code: code.into(), severity: sev.into(), region: reg.name.clone(), verts: n, detail });
        }
        let worst = pose_votes[r].iter().max_by_key(|(p, c)| (**c, u32::MAX - **p)).map(|(p, _)| ctx.poses[*p as usize].name.clone());
        regions.push(RegionReport {
            name: reg.name.clone(),
            label: reg.label.clone(),
            verts: verts[r],
            bad: bad[r],
            score: energy_to_score(energies[r]),
            energy: (energies[r] * 1e6).round() / 1e6,
            flags: flagc[r].iter().map(|(b, n)| (FLAG_NAMES.iter().find(|x| x.0 == *b).unwrap().1.to_string(), *n)).collect(),
            worst_pose: worst,
        });
    }
    if !model.seam_split.is_empty() {
        findings.push(Finding {
            code: "D_SEAM".into(),
            severity: "warn".into(),
            region: "*".into(),
            verts: model.seam_split.len(),
            detail: "UV-seam copies of a vertex carry different weights (cracks when posed)".into(),
        });
    }
    findings.sort_by(|a, b| (a.severity != "fail", &a.region, &a.code).cmp(&(b.severity != "fail", &b.region, &b.code)));
    regions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap().then(a.name.cmp(&b.name)));
    // Half the vertex mean, half the worst region: one broken limb or piece
    // must pull the score down even on a big mesh.
    let mean_e: f64 = ev.energy.iter().sum::<f64>() / model.nverts().max(1) as f64;
    let worst_e = energies.iter().cloned().fold(0.0, f64::max);
    let total_e = 0.5 * mean_e + 0.5 * worst_e;
    let mut worst_poses: Vec<PoseReport> =
        ev.pose_bad.iter().enumerate().map(|(i, &b)| PoseReport { pose: ctx.poses[i].name.clone(), bad: b }).collect();
    worst_poses.sort_by(|a, b| b.bad.cmp(&a.bad).then(a.pose.cmp(&b.pose)));
    worst_poses.truncate(12);
    Report {
        tool: "weightforge".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        file: file.into(),
        class: ctx.class.as_str().into(),
        verts: model.nverts(),
        tris: model.tris.len(),
        bones: model.njoints(),
        poses: ctx.poses.len(),
        scale: (model.scale * 1e6).round() / 1e6,
        score: energy_to_score(total_e),
        pass: !findings.iter().any(|f| f.severity == "fail"),
        findings,
        regions,
        worst_poses,
        thresholds: th.clone(),
    }
}

pub fn text(r: &Report) -> String {
    let mut s = String::new();
    let verdict = if r.pass { "PASS" } else { "FAIL" };
    s += &format!("{verdict} {} score {:.1}/100 ({}, {} verts, {} bones, {} poses)\n", r.file, r.score, r.class, r.verts, r.bones, r.poses);
    for f in &r.findings {
        s += &format!("{} {} [{}]: {} ({} verts)\n", f.code, f.region, f.severity, f.detail, f.verts);
    }
    let worst: Vec<&RegionReport> = r.regions.iter().filter(|x| x.bad > 0).take(8).collect();
    if !worst.is_empty() {
        s += "worst regions:\n";
        for x in worst {
            s += &format!("  {:<28} score {:>5.1}  bad {:>5}/{}\n", x.label, x.score, x.bad, x.verts);
        }
    }
    s
}

/// Per-vertex flags that belong to a reported finding (fail or warn), so a
/// sheet paints exactly what the report says and nothing below the
/// region threshold.
pub fn finding_mask(ctx: &Ctx, ev: &Eval, rep: &Report) -> Vec<u8> {
    let bit_of = |code: &str| -> u8 {
        match code {
            "D_STRETCH" => F_STRETCH,
            "D_VOLUME" => F_THIN,
            "D_BLEED" => F_BLEED,
            "D_NOISE" => F_NOISE,
            "D_INTERSECT" => F_INTERSECT,
            "D_UNWEIGHTED" => F_UNWEIGHTED,
            "D_PIECE_BONES" => F_PIECE,
            "D_PIECE_FOLLOW" => F_FOLLOW,
            _ => 0,
        }
    };
    let mut allowed: BTreeMap<&str, u8> = BTreeMap::new();
    for f in &rep.findings {
        *allowed.entry(f.region.as_str()).or_default() |= bit_of(&f.code);
    }
    (0..ev.flags.len())
        .map(|v| ev.flags[v] & allowed.get(ctx.regions[ctx.vregion[v] as usize].name.as_str()).copied().unwrap_or(0))
        .collect()
}
