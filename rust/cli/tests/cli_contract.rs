//! CLI contract: exit codes, outputs, and that --help lists every flag.

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_weights"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("wf_cli_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(bin()).args(args).output().unwrap();
    (o.status.code().unwrap_or(-1), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn exit_codes_and_outputs() {
    let d = tmp("contract");
    let ds = d.to_str().unwrap();
    assert_eq!(run(&["fixture", "all", "--out", ds]).0, 0);
    let clean = d.join("mannequin_clean.glb");
    let bleed = d.join("mannequin_bleed.glb");
    let (c, out, _) = run(&["check", clean.to_str().unwrap()]);
    assert_eq!(c, 0, "{out}");
    assert!(out.starts_with("PASS"));
    let (c, out, _) = run(&["check", bleed.to_str().unwrap(), "--json"]);
    assert_eq!(c, 1);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["pass"], false);
    assert!(v["findings"].as_array().unwrap().iter().any(|f| f["code"] == "D_BLEED"));

    let fixed = d.join("fixed.glb");
    let sheet = d.join("ab.png");
    let (c, out, _) = run(&["fix", bleed.to_str().unwrap(), "--out", fixed.to_str().unwrap(), "--sheet", sheet.to_str().unwrap()]);
    assert_eq!(c, 0, "{out}");
    assert!(out.starts_with("FIXED"));
    assert!(fixed.exists() && sheet.exists() && d.join("fixed.report.json").exists());
    assert_eq!(run(&["check", fixed.to_str().unwrap()]).0, 0);

    let dump = d.join("w.json");
    assert_eq!(run(&["dump", fixed.to_str().unwrap(), "--out", dump.to_str().unwrap()]).0, 0);
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&dump).unwrap()).unwrap();
    assert_eq!(v["joints"].as_array().unwrap().len(), 20);

    let png = d.join("s.png");
    assert_eq!(run(&["sheet", bleed.to_str().unwrap(), "--out", png.to_str().unwrap()]).0, 1);
    assert_eq!(&std::fs::read(&png).unwrap()[1..4], b"PNG");
    let cmp = d.join("c.png");
    assert_eq!(run(&["compare", bleed.to_str().unwrap(), fixed.to_str().unwrap(), "--out", cmp.to_str().unwrap()]).0, 0);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn usage_errors_exit_2() {
    assert_eq!(run(&[]).0, 2);
    assert_eq!(run(&["frobnicate"]).0, 2);
    assert_eq!(run(&["check"]).0, 2);
    assert_eq!(run(&["check", "/nonexistent.glb"]).0, 2);
    assert_eq!(run(&["fix", "x.glb"]).0, 2);
    assert_eq!(run(&["fix", "x.glb", "--out", "y.glb", "--method", "magic"]).0, 2);
    assert_eq!(run(&["check", "x.glb", "--voxels", "3"]).0, 2);
}

#[test]
fn help_lists_every_flag() {
    let (c, help, _) = run(&["--help"]);
    assert_eq!(c, 0);
    let src = include_str!("../src/main.rs");
    let mut flags: Vec<&str> = src.split('"').filter(|s| s.starts_with("--") && s.len() > 2 && !s.contains(' ')).collect();
    flags.sort();
    flags.dedup();
    for f in flags {
        assert!(help.contains(f), "--help does not mention {f}");
    }
}
