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
  weights fixture <fault|all> --out <dir>

COMMON OPTIONS:
  --poses <file.ron>  range-of-motion set (default: built-in by skeleton class)
  --no-clips          ignore the asset's own animation clips
  --voxels <N>        voxel resolution for geodesics (default 128)
  --json              machine-readable report on stdout

EXIT: 0 clean, 1 faults found, 2 usage/IO error.
";

struct Args {
    pos: Vec<String>,
    flags: Vec<(String, Option<String>)>,
}

const VALUED: &[&str] = &["--poses", "--voxels", "--out"];

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
    let (_, _, _, rep) = match wf::check(Path::new(input), &opts) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    if a.has("--json") {
        println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    } else {
        print!("{}", wf::report::text(&rep));
    }
    if rep.pass { ExitCode::SUCCESS } else { ExitCode::from(1) }
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
        "fixture" => cmd_fixture(&a),
        other => fail(format!("unknown command {other} (try --help)")),
    }
}
