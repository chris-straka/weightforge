//! `weights`: find and fix bad skin weights on rigged GLB characters.
//! Exit codes: 0 = clean / success, 1 = deformation faults found,
//! 2 = usage or IO error (same shape as rfcheck).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use weightforge_core as wf;
use wf::metrics::CtxOpts;

const HELP: &str = "\
weights — skin-weight QA and repair for rigged GLB characters

USAGE:
  weights check   <in.glb> [--poses rom.ron] [--json] [--no-clips] [--voxels N]
                  [--vertex-json flags.json]
  weights fix     <in.glb> --out <fixed.glb> [--method auto|smooth|geodesic|transfer|optimize]
                  [--source base.glb] [--candidate other.glb]... [--skintokens]
                  [--all-regions] [--sheet before_after.png]
  weights sheet   <in.glb> --out sheet.png [--poses rom.ron] [--cols N]
  weights compare <a.glb> <b.glb> --out ab.png [--poses rom.ron]
  weights dump    <in.glb> --out weights.json   (per-vertex joints/weights, for DCC import)
  weights fixture <fault|all> --out <dir>

FIX:
  Builds candidates (the input is always candidate 0), scores each on the
  same poses, keeps the best per failing region, blends at seams, and
  never writes anything worse than the input in any region. Writes
  <out>.report.json (or --report path) with before/after numbers.
  --method      auto (every method, default), or one or more of
                smooth,geodesic,transfer,optimize (comma-separated)
  --source      known-good rigged base to transfer weights from
  --candidate   extra weights to score (any rigged GLB of this mesh); repeatable
  --skintokens  also score SkinTokens weights for this skeleton (the ML
                candidate): runs `skintokens skin` ($SKINTOKENS_BIN, PATH, or
                ~/SWE/blender/skintokens/bin/skintokens; ~1 min on an M4).
                Never trusted blindly: scored like every other candidate
  --all-regions also change regions the check did not flag
  --sheet       write an A/B compare sheet (input vs fixed)

COMMON OPTIONS:
  --poses <file.ron>  range-of-motion set (default: built-in by skeleton class)
  --no-clips          ignore the asset's own animation clips
  --voxels <N>        voxel resolution for geodesics (default 128)
  --json              machine-readable report on stdout

  -h, --help          this text;  -V, --version  print the version

ENV: WF_DEBUG=1 traces fix's region trials on stderr. SKINTOKENS_BIN: see --skintokens.
EXIT: 0 clean (fix: output passes), 1 faults found, 2 usage/IO error.
";

struct Args {
    pos: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

const VALUED: &[&str] =
    &["--poses", "--voxels", "--out", "--cols", "--method", "--source", "--candidate", "--sheet", "--report", "--vertex-json"];

impl Args {
    fn parse(raw: Vec<String>) -> Result<Args, String> {
        let mut pos = Vec::new();
        let mut flags = Vec::new();
        let mut it = raw.into_iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                if VALUED.contains(&a.as_str()) {
                    let v = it.next().ok_or_else(|| format!("{a} needs a value"))?;
                    flags.push((a, Some(v)));
                } else {
                    flags.push((a, None));
                }
            } else {
                pos.push(a);
            }
        }
        Ok(Args { pos, flags })
    }
    fn has(&self, f: &str) -> bool {
        self.flags.iter().any(|(k, _)| k == f)
    }
    fn get(&self, f: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(k, _)| k == f).and_then(|(_, v)| v.as_deref())
    }
}

fn fail(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("weights: {msg}");
    ExitCode::from(2)
}

fn ctx_opts<'a>(a: &Args, set: Option<&'a wf::poses::PoseSet>) -> Result<CtxOpts<'a>, String> {
    let mut o = CtxOpts { poses: set, clips: !a.has("--no-clips"), ..Default::default() };
    if let Some(v) = a.get("--voxels") {
        o.voxel_res = v.parse().map_err(|_| "--voxels needs an integer".to_string())?;
        if !(16..=1024).contains(&o.voxel_res) {
            return Err("--voxels must be 16..1024".into());
        }
    }
    Ok(o)
}

fn load_poses(a: &Args) -> Result<Option<wf::poses::PoseSet>, String> {
    match a.get("--poses") {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
            wf::poses::parse_pose_set(&text).map(Some)
        }
        None => Ok(None),
    }
}

fn cmd_check(a: &Args) -> ExitCode {
    let Some(input) = a.pos.get(1) else { return fail("check needs an input .glb") };
    let set = match load_poses(a) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let opts = match ctx_opts(a, set.as_ref()) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let (model, ctx, ev, rep) = match wf::check(Path::new(input), &opts) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    if let Some(vj) = a.get("--vertex-json") {
        let mask = wf::report::finding_mask(&ctx, &ev, &rep);
        let meshes: Vec<serde_json::Value> = model
            .raw_meshes()
            .into_iter()
            .map(|(mesh, node, pos, map)| {
                serde_json::json!({
                    "mesh": mesh,
                    "node": node,
                    "positions": pos.iter().flatten().collect::<Vec<_>>(),
                    "flags": map.iter().map(|&w| mask[w as usize]).collect::<Vec<_>>(),
                    "energy": map.iter().map(|&w| (ev.energy[w as usize] * 1e4).round() / 1e4).collect::<Vec<_>>(),
                })
            })
            .collect();
        let flag_names: serde_json::Map<String, serde_json::Value> =
            wf::metrics::FLAG_NAMES.iter().map(|(b, n)| (n.to_string(), serde_json::json!(b))).collect();
        let doc = serde_json::json!({"flag_bits": flag_names, "meshes": meshes});
        if let Err(e) = std::fs::write(vj, serde_json::to_vec(&doc).unwrap()) {
            return fail(format!("{vj}: {e}"));
        }
    }
    if a.has("--json") {
        println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    } else {
        print!("{}", wf::report::text(&rep));
    }
    if rep.pass { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn fails(r: &wf::report::Report) -> usize {
    r.findings.iter().filter(|f| f.severity == "fail").count()
}

fn write_png(out: &str, img: &wf::render::Image) -> Result<(), String> {
    std::fs::write(out, img.png_bytes()).map_err(|e| format!("{out}: {e}"))
}

fn cmd_sheet(a: &Args) -> ExitCode {
    let Some(input) = a.pos.get(1) else { return fail("sheet needs an input .glb") };
    let Some(out) = a.get("--out") else { return fail("sheet needs --out <file.png>") };
    let set = match load_poses(a) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let opts = match ctx_opts(a, set.as_ref()) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let (model, ctx, ev, rep) = match wf::check(Path::new(input), &opts) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    let mut so = wf::render::SheetOpts::default();
    if let Some(c) = a.get("--cols") {
        match c.parse::<usize>() {
            Ok(n) if (1..=8).contains(&n) => so.cols = n,
            _ => return fail("--cols must be 1..8"),
        }
    }
    let mask = wf::report::finding_mask(&ctx, &ev, &rep);
    let info = wf::render::SheetInfo { title: &rep.file, mask: &mask, score: rep.score, pass: rep.pass, fails: fails(&rep) };
    let img = wf::render::sheet(&model, &ctx, &ev, &model.weights, &info, &so);
    if let Err(e) = write_png(out, &img) {
        return fail(e);
    }
    println!("{out}");
    if rep.pass { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn cmd_compare(a: &Args) -> ExitCode {
    let (Some(pa), Some(pb)) = (a.pos.get(1), a.pos.get(2)) else { return fail("compare needs two .glb files") };
    let Some(out) = a.get("--out") else { return fail("compare needs --out <file.png>") };
    let set = match load_poses(a) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let opts = match ctx_opts(a, set.as_ref()) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let ra = wf::check(Path::new(pa), &opts);
    let rb = wf::check(Path::new(pb), &opts);
    let ((ma, ca, ea, rpa), (mb, cb, eb, rpb)) = match (ra, rb) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) => return fail(e),
    };
    let mask_a = wf::report::finding_mask(&ca, &ea, &rpa);
    let mask_b = wf::report::finding_mask(&cb, &eb, &rpb);
    let sa = wf::render::Side {
        model: &ma,
        ctx: &ca,
        ev: &ea,
        w: &ma.weights,
        info: wf::render::SheetInfo { title: &rpa.file, mask: &mask_a, score: rpa.score, pass: rpa.pass, fails: fails(&rpa) },
    };
    let sb = wf::render::Side {
        model: &mb,
        ctx: &cb,
        ev: &eb,
        w: &mb.weights,
        info: wf::render::SheetInfo { title: &rpb.file, mask: &mask_b, score: rpb.score, pass: rpb.pass, fails: fails(&rpb) },
    };
    let img = wf::render::compare(&sa, &sb, &wf::render::SheetOpts::default());
    if let Err(e) = write_png(out, &img) {
        return fail(e);
    }
    println!("{out}");
    ExitCode::SUCCESS
}

/// The ML candidate: SkinTokens weights for the input's own skeleton
/// (`skintokens skin`, sibling repo ~/SWE/blender/skintokens). Err = the tool
/// is missing (usage error); Ok(None) = it ran and failed, so the fix goes on
/// without it (a warning, never a worse result).
fn skintokens_candidate(input: &str) -> Result<Option<wf::scene::Model>, String> {
    let bin = std::env::var_os("SKINTOKENS_BIN")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("skintokens")).find(|f| f.is_file())))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("SWE/blender/skintokens/bin/skintokens")).filter(|f| f.is_file())
        })
        .ok_or("skintokens not found (set SKINTOKENS_BIN or see ~/SWE/blender/skintokens)")?;
    if !bin.is_file() {
        return Err(format!("{} not found", bin.display()));
    }
    let dir = std::env::temp_dir().join(format!("weights_skintokens_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (out, report) = (dir.join("skintokens.glb"), dir.join("result.json"));
    eprintln!("weights: running SkinTokens on {input} (ML candidate)...");
    let status = std::process::Command::new(&bin)
        .arg("skin")
        .arg(input)
        .arg(&out)
        .arg("--report")
        .arg(&report)
        .stdout(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("{}: {e}", bin.display()))?;
    let model = if status.success() { wf::scene::Model::load(&out).ok() } else { None };
    if model.is_none() {
        let why = std::fs::read_to_string(&report).unwrap_or_default();
        eprintln!("weights: skintokens candidate skipped ({status}) {}", why.trim());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(model)
}

fn cmd_fix(a: &Args) -> ExitCode {
    let Some(input) = a.pos.get(1) else { return fail("fix needs an input .glb") };
    let Some(out) = a.get("--out") else { return fail("fix needs --out <fixed.glb>") };
    let methods = match wf::fix::Method::parse(a.get("--method").unwrap_or("auto")) {
        Some(m) => m,
        None => return fail("--method must be auto, smooth, geodesic, transfer, or optimize"),
    };
    let set = match load_poses(a) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    let opts = match ctx_opts(a, set.as_ref()) {
        Ok(o) => o,
        Err(e) => return fail(e),
    };
    let model = match wf::scene::Model::load(Path::new(input)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let source = match a.get("--source").map(|p| wf::scene::Model::load(Path::new(p))) {
        Some(Ok(m)) => Some(m),
        Some(Err(e)) => return fail(format!("--source: {e}")),
        None => None,
    };
    let mut externals = Vec::new();
    for (k, v) in &a.flags {
        if k == "--candidate" {
            let p = v.as_deref().unwrap_or("");
            match wf::scene::Model::load(Path::new(p)) {
                Ok(m) => externals.push((Path::new(p).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), m)),
                Err(e) => return fail(format!("--candidate {p}: {e}")),
            }
        }
    }
    if a.has("--skintokens") {
        match skintokens_candidate(input) {
            Ok(Some(m)) => externals.push(("skintokens".to_string(), m)),
            Ok(None) => {}
            Err(e) => return fail(format!("--skintokens: {e}")),
        }
    }
    let ctx = wf::metrics::Ctx::new(&model, &opts);
    let name = Path::new(input).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let fo = wf::fix::FixOpts {
        methods,
        source: source.as_ref(),
        external: externals.iter().map(|(n, m)| (n.clone(), m)).collect(),
        all_regions: a.has("--all-regions"),
    };
    let res = wf::fix::fix(&name, &model, &ctx, &fo);
    let glb = match model.write_weights(&res.weights) {
        Ok(g) => g,
        Err(e) => return fail(e),
    };
    if let Err(e) = std::fs::write(out, glb.to_bytes()) {
        return fail(format!("{out}: {e}"));
    }
    let report_path = a.get("--report").map(str::to_string).unwrap_or_else(|| {
        let p = Path::new(out);
        p.with_file_name(format!("{}.report.json", p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()))
            .to_string_lossy()
            .into_owned()
    });
    if let Err(e) = std::fs::write(&report_path, serde_json::to_string_pretty(&res.fix).unwrap()) {
        return fail(format!("{report_path}: {e}"));
    }
    if let Some(sheet_path) = a.get("--sheet") {
        let (rb, ra) = (&res.report_before, &res.report_after);
        let ev0 = wf::metrics::evaluate(&model, &ctx, &model.weights);
        let (ma, mb) = (wf::report::finding_mask(&ctx, &ev0, rb), wf::report::finding_mask(&ctx, &res.eval, ra));
        let ta = format!("{name} (input)");
        let tb = Path::new(out).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let sa = wf::render::Side {
            model: &model,
            ctx: &ctx,
            ev: &ev0,
            w: &model.weights,
            info: wf::render::SheetInfo { title: &ta, mask: &ma, score: rb.score, pass: rb.pass, fails: fails(rb) },
        };
        let sb = wf::render::Side {
            model: &model,
            ctx: &ctx,
            ev: &res.eval,
            w: &res.weights,
            info: wf::render::SheetInfo { title: &tb, mask: &mb, score: ra.score, pass: ra.pass, fails: fails(ra) },
        };
        if let Err(e) = write_png(sheet_path, &wf::render::compare(&sa, &sb, &wf::render::SheetOpts::default())) {
            return fail(e);
        }
    }
    let f = &res.fix;
    println!(
        "{} {name}: score {:.1} -> {:.1}, failing findings {} -> {}, {} verts changed",
        if f.after.pass {
            "FIXED"
        } else if f.improved {
            "IMPROVED"
        } else {
            "UNCHANGED"
        },
        f.before.score,
        f.after.score,
        f.before.fails,
        f.after.fails,
        f.verts_changed
    );
    for r in f.regions.iter().filter(|r| r.chosen != "original") {
        println!("  {:<24} {:>5.1} -> {:>5.1}  via {}", r.label, r.before, r.after, r.chosen);
    }
    println!("{out}\n{report_path}");
    if f.after.pass { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

fn cmd_dump(a: &Args) -> ExitCode {
    let Some(input) = a.pos.get(1) else { return fail("dump needs an input .glb") };
    let Some(out) = a.get("--out") else { return fail("dump needs --out <weights.json>") };
    let model = match wf::scene::Model::load(Path::new(input)) {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let meshes: Vec<serde_json::Value> = model
        .raw_meshes()
        .into_iter()
        .map(|(mesh, node, pos, map)| {
            serde_json::json!({
                "mesh": mesh,
                "node": node,
                "positions": pos.iter().flatten().collect::<Vec<_>>(),
                "influences": map.iter().map(|&w| model.weights[w as usize].iter().map(|&(j, x)| serde_json::json!([j, (x * 1e6).round() / 1e6])).collect::<Vec<_>>()).collect::<Vec<_>>(),
            })
        })
        .collect();
    let doc = serde_json::json!({"joints": model.skel.names, "meshes": meshes});
    if let Err(e) = std::fs::write(out, serde_json::to_vec(&doc).unwrap()) {
        return fail(format!("{out}: {e}"));
    }
    println!("{out}");
    ExitCode::SUCCESS
}

fn cmd_fixture(a: &Args) -> ExitCode {
    let Some(which) = a.pos.get(1) else { return fail("fixture needs a fault name or `all`") };
    let Some(out) = a.get("--out") else { return fail("fixture needs --out <dir>") };
    let names: Vec<&str> = if which == "all" {
        wf::fixtures::FAULTS.to_vec()
    } else if wf::fixtures::FAULTS.contains(&which.as_str()) {
        vec![which.as_str()]
    } else {
        return fail(format!("unknown fixture {which} (have: {})", wf::fixtures::FAULTS.join(", ")));
    };
    if let Err(e) = std::fs::create_dir_all(out) {
        return fail(format!("{out}: {e}"));
    }
    for n in names {
        let path = PathBuf::from(out).join(format!("mannequin_{n}.glb"));
        let glb = wf::fixtures::to_glb(&wf::fixtures::build(n));
        if let Err(e) = std::fs::write(&path, glb.to_bytes()) {
            return fail(format!("{}: {e}", path.display()));
        }
        println!("{}", path.display());
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw.iter().any(|a| a == "-h" || a == "--help") {
        print!("{HELP}");
        return if raw.is_empty() { ExitCode::from(2) } else { ExitCode::SUCCESS };
    }
    if raw.iter().any(|a| a == "-V" || a == "--version") {
        println!("weights {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let a = match Args::parse(raw) {
        Ok(a) => a,
        Err(e) => return fail(e),
    };
    match a.pos[0].as_str() {
        "check" => cmd_check(&a),
        "fix" => cmd_fix(&a),
        "sheet" => cmd_sheet(&a),
        "compare" => cmd_compare(&a),
        "dump" => cmd_dump(&a),
        "fixture" => cmd_fixture(&a),
        other => fail(format!("unknown command {other} (try --help)")),
    }
}
