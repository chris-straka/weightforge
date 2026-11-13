//! Bone roles from the skeleton's structure and the body's shape, not from
//! bone names.
//!
//! Names used to decide everything (`part_of` matched substrings, so the
//! `ring` of an eye-ring bone made it a finger, and a skeleton of `bone_N`
//! had no roles at all). Here a role comes from where a bone sits:
//!
//! - **feet** are the chains that end near the floor; joints within a few
//!   centimetres of each other end one foot (toes). Each foot's chain
//!   climbs to the last joint that holds no other foot: that is the leg.
//!   The heaviest legs (by the vertices they carry) are the legs; a skirt
//!   chain that reaches the floor carries less.
//! - two legs and an upright body: a **humanoid**. The hips join the legs.
//!   The spine runs from the hips to the bone that owns the top of the
//!   head; the two subtrees hanging off it that reach farthest to each
//!   side are the arms; the chest is where they hang. Head = the joint
//!   above the chest that owns the most vertices; neck = the ones between.
//! - in a leg: the thigh is the last joint still at hip height, the knee
//!   the joint halfway down to the ankle, the foot the joint at the bottom
//!   (and a last, mostly horizontal bone is the toe).
//! - in an arm (chain down to the hand, which is where the fingers branch,
//!   else the chain's end): upper arm and forearm are the pair of
//!   consecutive spans of similar length (0.7 to 1.5) that covers the most
//!   of the arm; the joint before them is the shoulder; everything under
//!   the hand is fingers.
//! - four legs: a **quadruped** (front pair = the pair ahead along the
//!   facing; hips and chest join each pair; neck and head beyond the
//!   chest; a tail behind the hips). Any other count of legs (a spider, a
//!   bird, a biped with a level spine): **legged**. No legs: **other**.
//!
//! Sides come from position (the character's left), never from the name.
//! Facing is +Z unless the feet's toe bones say otherwise.

use crate::math::{Vec3, v3};
use crate::scene::{Skeleton, Weights};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize)]
pub enum Body {
    Humanoid,
    Quadruped,
    Legged,
    #[default]
    Other,
}

impl Body {
    pub fn as_str(self) -> &'static str {
        match self {
            Body::Humanoid => "humanoid",
            Body::Quadruped => "quadruped",
            Body::Legged => "legged",
            Body::Other => "other",
        }
    }
}

/// What a joint belongs to, for telling a limb's own motion from another's.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default, serde::Serialize)]
pub enum Group {
    #[default]
    Torso,
    Head,
    Tail,
    /// A limb, named by its role prefix and side (`arm.L`, `leg.R`,
    /// `foreleg.L`, `leg3.R`).
    Limb(String),
}

impl Group {
    pub fn name(&self) -> String {
        match self {
            Group::Torso => "torso".into(),
            Group::Head => "head".into(),
            Group::Tail => "tail".into(),
            Group::Limb(s) => s.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Anatomy {
    pub body: Body,
    pub up: Vec3,
    pub forward: Vec3,
    /// Role -> joints (the chain's order for multi-bone roles): `hips`,
    /// `spine*`, `chest`, `neck`, `head`, `shoulder.L`, `upper_arm.L`,
    /// `forearm.L`, `hand.L`, `fingers.L`, `thigh.L`, `shin.L`, `foot.L`,
    /// `toe.L`, `tail*`; legged bodies `leg<k>_upper.L`, `leg<k>_lower.L`.
    pub roles: BTreeMap<String, Vec<usize>>,
    /// Per joint (helpers take their driver's).
    pub group: Vec<Group>,
    /// One line per thing found or missed, for the report.
    pub notes: Vec<String>,
}

struct Tree<'a> {
    sk: &'a Skeleton,
    parent: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    /// Vertices whose largest weight is on this joint (helpers: driver).
    mass: Vec<usize>,
    sub: Vec<usize>,
    body: Vec<bool>,
}

impl Tree<'_> {
    fn head(&self, j: usize) -> Vec3 {
        self.sk.head[j]
    }
    fn path_up(&self, mut j: usize, stop: usize) -> Vec<usize> {
        // j, parent(j), ... up to and excluding `stop`.
        let mut out = vec![];
        let mut guard = 0;
        while j != stop {
            out.push(j);
            match self.parent[j] {
                Some(p) => j = p,
                None => break,
            }
            guard += 1;
            if guard > self.parent.len() {
                break;
            }
        }
        out
    }
    fn ancestors(&self, j: usize) -> Vec<usize> {
        let mut out = vec![j];
        let mut p = self.parent[j];
        while let Some(q) = p {
            out.push(q);
            p = self.parent[q];
            if out.len() > self.parent.len() {
                break;
            }
        }
        out
    }
    fn lca(&self, a: usize, b: usize) -> Option<usize> {
        let aa = self.ancestors(a);
        self.ancestors(b).into_iter().find(|x| aa.contains(x))
    }
    fn descendants(&self, j: usize) -> Vec<usize> {
        let mut out = vec![];
        let mut stack = vec![j];
        while let Some(x) = stack.pop() {
            out.push(x);
            stack.extend(self.children[x].iter().copied());
        }
        out
    }
    fn is_desc(&self, j: usize, of: usize) -> bool {
        self.ancestors(j).contains(&of)
    }
}

fn build_tree<'a>(sk: &'a Skeleton, weights: &Weights) -> Tree<'a> {
    let n = sk.joints.len();
    let helper = |j: usize| crate::helpers::helper_of(sk, j).is_some();
    let mut parent = vec![None; n];
    for j in 0..n {
        if helper(j) {
            continue;
        }
        let mut p = sk.jparent[j];
        while let Some(q) = p {
            if !helper(q) {
                break;
            }
            p = sk.jparent[q];
        }
        parent[j] = p;
    }
    let mut mass = vec![0usize; n];
    for vw in weights {
        if let Some(&(j, w)) = vw.first() {
            if w > 0.0 && (j as usize) < n {
                mass[crate::helpers::owner(sk, j as usize)] += 1;
            }
        }
    }
    // Children lists, then subtree mass bottom-up (deepest first).
    let mut children = vec![Vec::new(); n];
    for j in 0..n {
        if let Some(p) = parent[j] {
            if !helper(j) {
                children[p].push(j);
            }
        }
    }
    let depth = |j: usize| -> usize {
        let mut d = 0;
        let mut p = parent[j];
        while let Some(q) = p {
            d += 1;
            p = parent[q];
            if d > n {
                break;
            }
        }
        d
    };
    let mut order: Vec<usize> = (0..n).filter(|&j| !helper(j)).collect();
    order.sort_by_key(|&j| std::cmp::Reverse(depth(j)));
    let mut sub = mass.clone();
    for &j in &order {
        if let Some(p) = parent[j] {
            sub[p] += sub[j];
        }
    }
    let body: Vec<bool> = (0..n).map(|j| !helper(j) && sub[j] > 0).collect();
    for c in &mut children {
        c.retain(|&x| body[x]);
        c.sort();
    }
    Tree { sk, parent, children, mass, sub, body }
}

/// The structural pass. `rest` and `weights` are the model's (welded).
pub fn analyze(sk: &Skeleton, rest: &[Vec3], weights: &Weights) -> Anatomy {
    let n = sk.joints.len();
    let t = build_tree(sk, weights);
    let up = v3(0.0, 1.0, 0.0);
    let mut a = Anatomy { up, forward: v3(0.0, 0.0, 1.0), group: vec![Group::Torso; n], ..Default::default() };
    if rest.is_empty() || n == 0 {
        a.notes.push("no mesh or no joints".into());
        return a;
    }
    let (mut g, mut top) = (f64::MAX, f64::MIN);
    for p in rest {
        g = g.min(p.y);
        top = top.max(p.y);
    }
    let h = (top - g).max(1e-9);
    let ht = |j: usize| t.head(j).y - g;
    // A leaf's lower end: its head, or its tail when that is lower (a
    // shin whose unweighted end bone stands on the floor).
    let low_end = |j: usize| if sk.tail[j].y < t.head(j).y { sk.tail[j] } else { t.head(j) };

    // ---- feet: body leaves near the floor, clustered into feet.
    let leaves: Vec<usize> = (0..n).filter(|&j| t.body[j] && t.children[j].is_empty()).collect();
    // A leaf owning almost nothing (a stray bone at the origin) is no foot.
    let min_mass = (weights.len() / 500).max(1);
    let mut low: Vec<usize> = leaves.iter().copied().filter(|&j| low_end(j).y - g < 0.15 * h && t.mass[j] >= min_mass).collect();
    let horiz = |p: Vec3| (p.x * p.x + p.z * p.z).sqrt();
    // Feet: the heaviest low leaf seeds a foot; leaves within 3.5% of the
    // height of a seed join it (toes of one foot).
    low.sort_by_key(|&j| (std::cmp::Reverse(t.mass[j]), j));
    let mut seeds: Vec<(usize, Vec<usize>)> = Vec::new();
    for &j in &low {
        // Same foot: close together, or meeting low down (a heel and a toe
        // under one foot bone, the digits of a paw).
        let same = |a: usize, b: usize| {
            horiz(low_end(a) - low_end(b)) < 0.035 * h
                || t.lca(a, b)
                    .is_some_and(|c| ht(c) < 0.2 * h && (t.head(c) - low_end(a)).len().max((t.head(c) - low_end(b)).len()) < 0.2 * h)
        };
        match seeds.iter_mut().find(|(s0, _)| same(*s0, j)) {
            Some((_, m)) => m.push(j),
            None => seeds.push((j, vec![j])),
        }
    }
    let feet: BTreeMap<usize, Vec<usize>> = seeds.into_iter().collect();
    // Each foot: the deepest leaf ends its chain; the chain climbs while
    // no other foot hangs below.
    struct Leg {
        root: usize,
        end: usize,
        chain: Vec<usize>,
        mass: usize,
        foot_pos: Vec3,
    }
    let foot_list: Vec<Vec<usize>> = feet.into_values().collect();
    let mut legs: Vec<Leg> = Vec::new();
    for (fi, f) in foot_list.iter().enumerate() {
        let others: Vec<usize> = foot_list.iter().enumerate().filter(|(k, _)| *k != fi).flat_map(|(_, x)| x.iter().copied()).collect();
        // Each leaf of the foot climbs on its own (an IK bone left on the
        // floor under the root climbs nowhere); the foot's leg is the
        // heaviest of those climbs.
        let mut best: Option<Leg> = None;
        for &end in f {
            let mut root = end;
            while let Some(p) = t.parent[root] {
                if others.iter().any(|&o| t.is_desc(o, p)) {
                    break;
                }
                root = p;
            }
            // A root that holds the whole skeleton (one foot alone) is no leg.
            if t.parent[root].is_none() {
                continue;
            }
            let mut chain = t.path_up(end, root);
            chain.push(root);
            chain.reverse();
            let foot_pos = low_end(end);
            let long = ht(root) - (foot_pos.y - g) > 0.18 * h;
            if long && chain.len() >= 2 && best.as_ref().is_none_or(|b| t.sub[root] > b.mass) {
                best = Some(Leg { root, end, chain, mass: t.sub[root], foot_pos });
            }
        }
        if let Some(l) = best {
            if !legs.iter().any(|x| x.root == l.root) {
                legs.push(l);
            }
        }
    }
    if std::env::var_os("WF_DEBUG_ANATOMY").is_some() {
        let nm = |j: usize| sk.names[j].clone();
        eprintln!(
            "anatomy: ground {g:.3} height {h:.3}; low leaves {:?}",
            low.iter().map(|&j| (nm(j), (low_end(j).y - g) / h)).collect::<Vec<_>>()
        );
        for f in &foot_list {
            eprintln!("  foot {:?}", f.iter().map(|&j| nm(j)).collect::<Vec<_>>());
        }
        for l in &legs {
            eprintln!("  leg root {} mass {} chain {:?}", nm(l.root), l.mass, l.chain.iter().map(|&j| nm(j)).collect::<Vec<_>>());
        }
    }
    // The heaviest legs; a floor-length skirt chain carries less.
    let max_mass = legs.iter().map(|l| l.mass).max().unwrap_or(0);
    legs.retain(|l| l.mass * 4 >= max_mass && l.mass > 0);
    legs.sort_by_key(|l| (std::cmp::Reverse(l.mass), l.root));
    if legs.len() > 4 && legs.len() % 2 == 1 {
        legs.pop();
    }
    // ---- facing from the toes (last bone of a leg, mostly level).
    let mut fsum = Vec3::ZERO;
    for l in &legs {
        let c = &l.chain;
        if c.len() >= 2 {
            let d = t.head(c[c.len() - 1]) - t.head(c[c.len() - 2]);
            let flat = v3(d.x, 0.0, d.z);
            if flat.len() > d.y.abs() && flat.len() > 1e-9 {
                fsum += flat.normalized();
            }
        }
    }
    if fsum.len() > 0.5 {
        // Characters stand on an axis: snap to the nearest one.
        let f = fsum.normalized();
        a.forward = if f.x.abs() > f.z.abs() { v3(f.x.signum(), 0.0, 0.0) } else { v3(0.0, 0.0, f.z.signum()) };
    }
    let left = up.cross(a.forward).normalized();
    let side_of = |p: Vec3, c: Vec3| if (p - c).dot(left) >= 0.0 { "L" } else { "R" };

    let put = |a: &mut Anatomy, role: String, js: Vec<usize>| {
        if !js.is_empty() {
            a.roles.insert(role, js);
        }
    };
    let leg_roles = |a: &mut Anatomy, l: &Leg, prefix_upper: &str, prefix_lower: &str, prefix_foot: &str, prefix_toe: &str, side: &str| {
        let c = &l.chain;
        // Toe: a last, mostly level bone.
        let mut foot_i = c.len() - 1;
        if c.len() >= 3 {
            let d = t.head(c[c.len() - 1]) - t.head(c[c.len() - 2]);
            if v3(d.x, 0.0, d.z).len() > d.y.abs() {
                a.roles.insert(format!("{prefix_toe}.{side}"), vec![c[c.len() - 1]]);
                foot_i = c.len() - 2;
            }
        }
        let yfoot = t.head(c[foot_i]).y;
        let y0 = t.head(c[0]).y;
        let drop = (y0 - yfoot).max(1e-9);
        // Thigh: the last joint still at hip height.
        let mut ti = 0;
        for (i, &j) in c.iter().enumerate().take(foot_i) {
            if t.head(j).y >= y0 - 0.15 * drop {
                ti = i;
            }
        }
        let ymid = 0.5 * (t.head(c[ti]).y + yfoot);
        let span = (t.head(c[ti]).y - yfoot).max(1e-9);
        let knee = (ti + 1..foot_i)
            .filter(|&i| (t.head(c[i]).y - ymid).abs() < 0.3 * span)
            .min_by(|&x, &y| (t.head(c[x]).y - ymid).abs().partial_cmp(&(t.head(c[y]).y - ymid).abs()).unwrap());
        if foot_i > ti {
            a.roles.insert(format!("{prefix_upper}.{side}"), vec![c[ti]]);
            if let Some(k) = knee {
                a.roles.insert(format!("{prefix_lower}.{side}"), vec![c[k]]);
            }
            a.roles.insert(format!("{prefix_foot}.{side}"), vec![c[foot_i]]);
        }
        let _ = l.end;
    };
    let set_group = |a: &mut Anatomy, root: usize, g: Group| {
        for j in t.descendants(root) {
            a.group[j] = g.clone();
        }
    };

    if legs.len() == 2 {
        let hips = t.lca(legs[0].root, legs[1].root);
        let crown = crown_joint(&t, rest, weights, top, h, left);
        let upright = match (hips, crown) {
            (Some(hp), Some(c)) => t.is_desc(c, hp) && ht(c) - ht(hp) > 0.2 * h,
            _ => false,
        };
        if let (true, Some(hp), Some(crown)) = (upright, hips, crown) {
            humanoid(&mut a, &t, &legs.iter().map(|l| (l.root, l.foot_pos)).collect::<Vec<_>>(), hp, crown, left, h, &mut |a, k| {
                let l = &legs[k];
                let side = side_of(l.foot_pos, t.head(hp));
                leg_roles(a, l, "thigh", "shin", "foot", "toe", side);
                set_group(a, l.root, Group::Limb(format!("leg.{side}")));
            });
            if a.body == Body::Humanoid {
                return finish(a, sk);
            }
        }
    }
    if legs.len() == 4 {
        // Front pair: ahead along the facing.
        let mut idx: Vec<usize> = (0..4).collect();
        idx.sort_by(|&x, &y| legs[y].foot_pos.dot(a.forward).partial_cmp(&legs[x].foot_pos.dot(a.forward)).unwrap());
        let (front, back) = ([idx[0], idx[1]], [idx[2], idx[3]]);
        let chest = t.lca(legs[front[0]].root, legs[front[1]].root);
        let hips = t.lca(legs[back[0]].root, legs[back[1]].root);
        if let (Some(ch), Some(hp)) = (chest, hips) {
            a.body = Body::Quadruped;
            for (pair, names, grp) in
                [(back, ["thigh", "shin", "foot", "toe"], "leg"), (front, ["upper_arm", "forearm", "hand", "fingers"], "foreleg")]
            {
                for &k in &pair {
                    let l = &legs[k];
                    let side = side_of(l.foot_pos, t.head(if grp == "leg" { hp } else { ch }));
                    leg_roles(&mut a, l, names[0], names[1], names[2], names[3], side);
                    set_group(&mut a, l.root, Group::Limb(format!("{grp}.{side}")));
                }
            }
            put(&mut a, "hips".into(), vec![hp]);
            // Spine: hips to chest along the tree (either may hold the other).
            let spine: Vec<usize> = if t.is_desc(ch, hp) {
                let mut p = t.path_up(ch, hp);
                p.reverse();
                p
            } else if t.is_desc(hp, ch) {
                t.path_up(hp, ch)
            } else {
                vec![ch]
            };
            put(&mut a, "spine*".into(), spine);
            put(&mut a, "chest".into(), vec![ch]);
            let leg_roots: Vec<usize> = legs.iter().map(|l| l.root).collect();
            // Neck and head: the heaviest chain out of the chest that is no
            // leg, ahead of the chest; tail: the one behind the hips.
            let fwd = a.forward;
            let reach = |r: usize, dir: Vec3| t.descendants(r).iter().map(|&j| (t.head(j) - t.head(r)).dot(dir)).fold(0.0, f64::max);
            let off = |base: usize| -> Vec<usize> {
                t.children[base].iter().copied().filter(|c| !leg_roots.contains(c) && !t.is_desc(hp, *c) && !t.is_desc(ch, *c)).collect()
            };
            if let Some(nk) = off(ch).into_iter().filter(|&c| reach(c, fwd) > 0.05 * h).max_by_key(|&c| (t.sub[c], c)) {
                let chain = main_chain(&t, nk);
                let head = *chain.iter().max_by_key(|&&j| (t.mass[j], j)).unwrap();
                let hi = chain.iter().position(|&j| j == head).unwrap();
                put(&mut a, "neck".into(), chain[..hi].to_vec());
                put(&mut a, "head".into(), vec![head]);
                set_group(&mut a, nk, Group::Head);
            }
            if let Some(tl) = off(hp).into_iter().filter(|&c| reach(c, -fwd) > 0.05 * h).max_by_key(|&c| (t.sub[c], c)) {
                put(&mut a, "tail*".into(), main_chain(&t, tl));
                set_group(&mut a, tl, Group::Tail);
            }
            a.notes.push("four legs: quadruped".into());
            return finish(a, sk);
        }
    }
    if !legs.is_empty() {
        a.body = Body::Legged;
        let centre = legs.iter().map(|l| l.foot_pos).fold(Vec3::ZERO, |s, p| s + p) / legs.len() as f64;
        let mut order: Vec<usize> = (0..legs.len()).collect();
        order.sort_by(|&x, &y| legs[y].foot_pos.dot(a.forward).partial_cmp(&legs[x].foot_pos.dot(a.forward)).unwrap().then(x.cmp(&y)));
        let mut count: BTreeMap<&str, usize> = BTreeMap::new();
        for &k in &order {
            let l = &legs[k];
            let side = side_of(l.foot_pos, centre);
            let i = *count.entry(side).and_modify(|c| *c += 1).or_insert(0);
            let c = &l.chain;
            put(&mut a, format!("leg{i}_upper.{side}"), vec![c[0]]);
            if c.len() >= 3 {
                put(&mut a, format!("leg{i}_lower.{side}"), vec![c[c.len() / 2]]);
            }
            set_group(&mut a, l.root, Group::Limb(format!("leg{i}.{side}")));
        }
        a.notes.push(format!("{} legs: legged body", legs.len()));
        return finish(a, sk);
    }
    a.body = Body::Other;
    a.notes.push("no legs found (no chain ends near the floor with weight on it)".into());
    finish(a, sk)
}

/// A name split into lower-case words (`DEF-upper_arm.L`, `LeftShoulder`
/// -> upper, arm, l / left, shoulder): `true` if any word is in `words`.
fn has_token(name: &str, words: &[&str]) -> bool {
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;
    for c in name.chars() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                toks.push(std::mem::take(&mut cur));
            }
            prev_lower = false;
            continue;
        }
        if c.is_uppercase() && prev_lower && !cur.is_empty() {
            toks.push(std::mem::take(&mut cur));
        }
        prev_lower = c.is_lowercase();
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks.iter().any(|t| words.contains(&t.trim_end_matches(|c: char| c.is_ascii_digit())))
}

/// Helpers take their driver's group.
fn finish(mut a: Anatomy, sk: &Skeleton) -> Anatomy {
    for h in &sk.helpers {
        a.group[h.joint] = a.group[h.driver].clone();
    }
    a
}

/// Follows the heaviest child from `j` to a leaf.
fn main_chain(t: &Tree, mut j: usize) -> Vec<usize> {
    let mut out = vec![j];
    while let Some(&c) = t.children[j].iter().max_by_key(|&&c| (t.sub[c], std::cmp::Reverse(c))) {
        out.push(c);
        j = c;
        if out.len() > t.parent.len() {
            break;
        }
    }
    out
}

/// The joint owning the top of the body near its midline (the crown): the
/// topmost vertices near the joints' midline vote with their largest
/// weight's joint.
fn crown_joint(t: &Tree, rest: &[Vec3], weights: &Weights, top: f64, h: f64, left: Vec3) -> Option<usize> {
    let body: Vec<usize> = (0..t.body.len()).filter(|&j| t.body[j]).collect();
    let mid = body.iter().map(|&j| t.head(j).dot(left)).sum::<f64>() / body.len().max(1) as f64;
    let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
    for (p, vw) in rest.iter().zip(weights) {
        if p.y > top - 0.04 * h && (p.dot(left) - mid).abs() < 0.1 * h {
            if let Some(&(j, _)) = vw.first() {
                *votes.entry(crate::helpers::owner(t.sk, j as usize)).or_default() += 1;
            }
        }
    }
    votes.into_iter().max_by_key(|&(j, m)| (m, std::cmp::Reverse(j))).map(|(j, _)| j)
}

#[allow(clippy::too_many_arguments)]
fn humanoid(
    a: &mut Anatomy,
    t: &Tree,
    legs: &[(usize, Vec3)],
    hips: usize,
    crown: usize,
    left: Vec3,
    h: f64,
    leg_roles: &mut dyn FnMut(&mut Anatomy, usize),
) {
    // Spine: hips up to the crown.
    let mut path = t.path_up(crown, hips);
    path.push(hips);
    path.reverse();
    let leg_roots: Vec<usize> = legs.iter().map(|l| l.0).collect();
    // Arms: per side, the subtree off the spine path (not a leg, not the
    // path) reaching farthest that way.
    let mid = t.head(hips).dot(left);
    let mut arms: [Option<(usize, usize, f64)>; 2] = [None, None]; // (root, attach index, reach)
    for (pi, &p) in path.iter().enumerate() {
        for &c in &t.children[p] {
            if path.contains(&c) || leg_roots.contains(&c) {
                continue;
            }
            let d = t.descendants(c);
            for (si, sgn) in [(0usize, 1.0f64), (1, -1.0)] {
                let reach = d.iter().map(|&j| sgn * (t.head(j).dot(left) - mid)).fold(f64::MIN, f64::max);
                if reach > 0.12 * h && arms[si].is_none_or(|x| reach > x.2) {
                    arms[si] = Some((c, pi, reach));
                }
            }
        }
    }
    let (Some(al), Some(ar)) = (arms[0], arms[1]) else {
        a.notes.push("two legs, but no arm reaching each side: not a humanoid".into());
        return;
    };
    if al.0 == ar.0 {
        a.notes.push("two legs, one subtree reaching both sides: not a humanoid".into());
        return;
    }
    a.body = Body::Humanoid;
    for k in 0..legs.len() {
        leg_roles(a, k);
    }
    let chest_i = al.1.max(ar.1).max(1).min(path.len() - 1);
    a.roles.insert("hips".into(), vec![hips]);
    a.roles.insert("spine*".into(), path[1..=chest_i].to_vec());
    a.roles.insert("chest".into(), vec![path[chest_i]]);
    // Head: the joint above the chest owning the most vertices.
    let above: Vec<usize> = path[chest_i + 1..].to_vec();
    if let Some(&head) = above.iter().max_by_key(|&&j| (t.mass[j], std::cmp::Reverse(j))) {
        let hi = above.iter().position(|&j| j == head).unwrap();
        if hi > 0 {
            a.roles.insert("neck".into(), above[..hi].to_vec());
        }
        a.roles.insert("head".into(), vec![head]);
        for j in t.descendants(above[0]) {
            a.group[j] = Group::Head;
        }
    }
    for (root, _, _) in [al, ar] {
        let side = if t
            .head(
                *t.descendants(root)
                    .iter()
                    .max_by(|&&x, &&y| (t.head(x).dot(left) - mid).abs().partial_cmp(&(t.head(y).dot(left) - mid).abs()).unwrap())
                    .unwrap(),
            )
            .dot(left)
            >= mid
        {
            "L"
        } else {
            "R"
        };
        arm_roles(a, t, root, side, left, mid, h);
        for j in t.descendants(root) {
            a.group[j] = Group::Limb(format!("arm.{side}"));
        }
    }
    a.notes.push("two legs and two arms: humanoid".into());
}

fn arm_roles(a: &mut Anatomy, t: &Tree, root: usize, side: &str, left: Vec3, mid: f64, h: f64) {
    // The arm's chain: from its root to the joint reaching farthest (a
    // fingertip, or the hand of a mitten).
    let far = *t
        .descendants(root)
        .iter()
        .max_by(|&&x, &&y| (t.head(x) - t.head(root)).len().partial_cmp(&(t.head(y) - t.head(root)).len()).unwrap().then(y.cmp(&x)))
        .unwrap();
    let mut p = t.path_up(far, root);
    p.push(root);
    p.reverse();
    let reach = (t.head(far) - t.head(root)).len().max(1e-9);
    // Hand: the first joint past 72% of that reach (the hand and fingers
    // are the last quarter of an arm).
    let hand_i = (1..p.len()).find(|&i| (t.head(p[i]) - t.head(root)).len() >= 0.72 * reach).unwrap_or(p.len() - 1);
    let s: Vec<f64> = {
        let mut acc = vec![0.0];
        for w in p.windows(2) {
            acc.push(acc.last().unwrap() + (t.head(w[1]) - t.head(w[0])).len());
        }
        acc
    };
    // Upper arm: after a joint the rig calls a clavicle or shoulder (a
    // whole-word hint inside a chain found by structure), else the first
    // joint clear of the midline (a clavicle starts at it), with room for
    // a forearm before the hand.
    let lateral = |j: usize| (t.head(j).dot(left) - mid).abs();
    let collar = (0..hand_i.saturating_sub(1))
        .filter(|&i| has_token(&t.sk.names[p[i]], &["shoulder", "clavicle", "collar", "collarbone", "scapula"]))
        .last();
    let ui = match collar {
        Some(c) => Some(c + 1),
        None => (0..hand_i.saturating_sub(1)).find(|&i| lateral(p[i]) >= 0.055 * h).or(hand_i.checked_sub(2)),
    };
    // Elbow: the joint nearest the middle of the arc from there to the hand
    // (a twist split stays inside a span).
    let elbow = ui.and_then(|ui| {
        let m = 0.5 * (s[ui] + s[hand_i]);
        let span = (s[hand_i] - s[ui]).max(1e-9);
        (ui + 1..hand_i)
            .filter(|&k| (s[k] - m).abs() < 0.3 * span)
            .min_by(|&x, &y| (s[x] - m).abs().partial_cmp(&(s[y] - m).abs()).unwrap())
    });
    let best = ui.zip(elbow);
    a.roles.insert(format!("hand.{side}"), vec![p[hand_i]]);
    let fingers: Vec<usize> = t.descendants(p[hand_i]).into_iter().filter(|&j| j != p[hand_i]).collect();
    if !fingers.is_empty() {
        let mut f = fingers;
        f.sort();
        a.roles.insert(format!("fingers.{side}"), f);
    }
    match best {
        Some((i, k)) => {
            a.roles.insert(format!("upper_arm.{side}"), vec![p[i]]);
            a.roles.insert(format!("forearm.{side}"), vec![p[k]]);
            if i > 0 {
                a.roles.insert(format!("shoulder.{side}"), vec![p[i - 1]]);
            }
        }
        None => a.notes.push(format!("arm.{side}: no elbow between the shoulder and the hand")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    fn model(f: &fixtures::Fixture) -> crate::scene::Model {
        crate::scene::Model::from_glb(fixtures::to_glb(f)).unwrap()
    }

    #[test]
    fn mannequin_roles_match_its_names() {
        let m = model(&fixtures::build("clean"));
        let a = &m.skel.anatomy;
        assert_eq!(a.body, Body::Humanoid, "{:?}", a.notes);
        let name = |r: &str| a.roles.get(r).map(|js| js.iter().map(|&j| m.skel.names[j].clone()).collect::<Vec<_>>());
        assert_eq!(name("hips").unwrap(), ["DEF-spine"]);
        assert_eq!(name("spine*").unwrap(), ["DEF-spine.001", "DEF-spine.002", "DEF-spine.003"]);
        assert_eq!(name("neck").unwrap(), ["DEF-neck"]);
        assert_eq!(name("head").unwrap(), ["DEF-head"]);
        for s in ["L", "R"] {
            assert_eq!(name(&format!("shoulder.{s}")).unwrap(), [format!("DEF-shoulder.{s}")]);
            assert_eq!(name(&format!("upper_arm.{s}")).unwrap(), [format!("DEF-upper_arm.{s}")]);
            assert_eq!(name(&format!("forearm.{s}")).unwrap(), [format!("DEF-forearm.{s}")]);
            assert_eq!(name(&format!("hand.{s}")).unwrap(), [format!("DEF-hand.{s}")]);
            assert_eq!(name(&format!("thigh.{s}")).unwrap(), [format!("DEF-thigh.{s}")]);
            assert_eq!(name(&format!("shin.{s}")).unwrap(), [format!("DEF-shin.{s}")]);
            assert_eq!(name(&format!("foot.{s}")).unwrap(), [format!("DEF-foot.{s}")]);
        }
        assert_eq!(a.group[m.skel.joint_by_name("DEF-hand.L").unwrap()], Group::Limb("arm.L".into()));
        assert_eq!(a.group[m.skel.joint_by_name("DEF-shin.R").unwrap()], Group::Limb("leg.R".into()));
        assert_eq!(a.group[m.skel.joint_by_name("DEF-head").unwrap()], Group::Head);
        assert_eq!(a.group[m.skel.joint_by_name("DEF-spine.002").unwrap()], Group::Torso);
    }

    #[test]
    fn names_do_not_matter() {
        // Every bone renamed bone_N, and sides swapped in the names: roles
        // and sides come out the same.
        let mut f = fixtures::build("clean");
        let real: Vec<String> = f.bones.iter().map(|b| b.name.to_string()).collect();
        for (i, b) in f.bones.iter_mut().enumerate() {
            b.name = Box::leak(format!("bone_{i}{}", if i % 3 == 0 { "_ring.R" } else { "" }).into_boxed_str());
        }
        let m = model(&f);
        let a = &m.skel.anatomy;
        assert_eq!(a.body, Body::Humanoid, "{:?}", a.notes);
        let real_of = |r: &str| a.roles.get(r).map(|js| js.iter().map(|&j| real[j].clone()).collect::<Vec<_>>());
        assert_eq!(real_of("forearm.L").unwrap(), ["DEF-forearm.L"]);
        assert_eq!(real_of("shin.R").unwrap(), ["DEF-shin.R"]);
        assert_eq!(real_of("head").unwrap(), ["DEF-head"]);
        assert!(a.roles.keys().all(|k| !k.starts_with("fingers")), "{:?}", a.roles.keys());
    }
}
