//! Small f64 linear algebra: just what skinning, metrics, and rendering need.
//! Own code (no SIMD crates) keeps results byte-deterministic per platform.

use std::ops::{Add, AddAssign, Div, Index, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);
    pub fn from_f32(a: [f32; 3]) -> Vec3 {
        v3(a[0] as f64, a[1] as f64, a[2] as f64)
    }
    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        v3(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    pub fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    pub fn len2(self) -> f64 {
        self.dot(self)
    }
    pub fn normalized(self) -> Vec3 {
        let l = self.len();
        if l > 1e-12 { self / l } else { Vec3::ZERO }
    }
    pub fn min(self, o: Vec3) -> Vec3 {
        v3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    pub fn max(self, o: Vec3) -> Vec3 {
        v3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    pub fn lerp(self, o: Vec3, t: f64) -> Vec3 {
        self + (o - self) * t
    }
    /// Any unit vector perpendicular to self (self must be non-zero).
    pub fn any_perp(self) -> Vec3 {
        let a = if self.x.abs() < 0.9 { v3(1.0, 0.0, 0.0) } else { v3(0.0, 1.0, 0.0) };
        self.cross(a).normalized()
    }
    pub fn max_elem(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        v3(self.x / s, self.y / s, self.z / s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}
impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

/// Unit quaternion (x, y, z, w).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Quat {
        let a = axis.normalized();
        let (s, c) = (angle * 0.5).sin_cos();
        Quat { x: a.x * s, y: a.y * s, z: a.z * s, w: c }
    }
    /// Shortest rotation taking unit vector `a` onto unit vector `b`.
    pub fn from_to(a: Vec3, b: Vec3) -> Quat {
        let d = a.dot(b).clamp(-1.0, 1.0);
        if d > 1.0 - 1e-12 {
            return Quat::IDENTITY;
        }
        if d < -1.0 + 1e-12 {
            return Quat::from_axis_angle(a.any_perp(), std::f64::consts::PI);
        }
        Quat::from_axis_angle(a.cross(b), d.acos())
    }
    pub fn normalized(self) -> Quat {
        let l = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if l < 1e-15 {
            return Quat::IDENTITY;
        }
        Quat { x: self.x / l, y: self.y / l, z: self.z / l, w: self.w / l }
    }
    pub fn conj(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }
    pub fn dot(self, o: Quat) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = v3(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }
    pub fn slerp(self, mut o: Quat, t: f64) -> Quat {
        let mut d = self.dot(o);
        if d < 0.0 {
            o = Quat { x: -o.x, y: -o.y, z: -o.z, w: -o.w };
            d = -d;
        }
        if d > 0.9995 {
            return Quat {
                x: self.x + (o.x - self.x) * t,
                y: self.y + (o.y - self.y) * t,
                z: self.z + (o.z - self.z) * t,
                w: self.w + (o.w - self.w) * t,
            }
            .normalized();
        }
        let th = d.acos();
        let s = th.sin();
        let a = ((1.0 - t) * th).sin() / s;
        let b = (t * th).sin() / s;
        Quat { x: self.x * a + o.x * b, y: self.y * a + o.y * b, z: self.z * a + o.z * b, w: self.w * a + o.w * b }
    }
}

impl Mul for Quat {
    type Output = Quat;
    fn mul(self, o: Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }
}

/// Column-major 4x4 affine matrix (glTF layout): `m[col][row]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 { m: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
    pub fn from_cols_slice(a: &[f64]) -> Mat4 {
        let mut m = [[0.0; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                m[c][r] = a[c * 4 + r];
            }
        }
        Mat4 { m }
    }
    pub fn to_cols_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(16);
        for c in 0..4 {
            for r in 0..4 {
                v.push(self.m[c][r]);
            }
        }
        v
    }
    pub fn from_trs(t: Vec3, r: Quat, s: Vec3) -> Mat4 {
        let (x, y, z, w) = (r.x, r.y, r.z, r.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Mat4 {
            m: [
                [(1.0 - (yy + zz)) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0.0],
                [(xy - wz) * s.y, (1.0 - (xx + zz)) * s.y, (yz + wx) * s.y, 0.0],
                [(xz + wy) * s.z, (yz - wx) * s.z, (1.0 - (xx + yy)) * s.z, 0.0],
                [t.x, t.y, t.z, 1.0],
            ],
        }
    }
    pub fn translation(&self) -> Vec3 {
        v3(self.m[3][0], self.m[3][1], self.m[3][2])
    }
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let m = &self.m;
        v3(
            m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
            m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
            m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
        )
    }
    pub fn transform_vector(&self, p: Vec3) -> Vec3 {
        let m = &self.m;
        v3(
            m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z,
            m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z,
            m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z,
        )
    }
    pub fn scale_add(&mut self, o: &Mat4, s: f64) {
        for c in 0..4 {
            for r in 0..4 {
                self.m[c][r] += o.m[c][r] * s;
            }
        }
    }
    pub fn zero() -> Mat4 {
        Mat4 { m: [[0.0; 4]; 4] }
    }
    /// General inverse (affine matrices in practice). Returns identity if singular.
    pub fn inverse(&self) -> Mat4 {
        // Gauss-Jordan on row-major copy.
        let mut a = [[0.0f64; 8]; 4];
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] = self.m[c][r];
            }
            a[r][4 + r] = 1.0;
        }
        for col in 0..4 {
            let mut piv = col;
            for r in col + 1..4 {
                if a[r][col].abs() > a[piv][col].abs() {
                    piv = r;
                }
            }
            if a[piv][col].abs() < 1e-14 {
                return Mat4::IDENTITY;
            }
            a.swap(col, piv);
            let d = a[col][col];
            for c in 0..8 {
                a[col][c] /= d;
            }
            for r in 0..4 {
                if r != col {
                    let f = a[r][col];
                    if f != 0.0 {
                        for c in 0..8 {
                            a[r][c] -= f * a[col][c];
                        }
                    }
                }
            }
        }
        let mut out = Mat4::zero();
        for r in 0..4 {
            for c in 0..4 {
                out.m[c][r] = a[r][4 + c];
            }
        }
        out
    }
    /// Rotation part as a quaternion (assumes no shear; scale is divided out).
    pub fn rotation(&self) -> Quat {
        let cx = v3(self.m[0][0], self.m[0][1], self.m[0][2]).normalized();
        let cy = v3(self.m[1][0], self.m[1][1], self.m[1][2]).normalized();
        let cz = v3(self.m[2][0], self.m[2][1], self.m[2][2]).normalized();
        quat_from_basis(cx, cy, cz)
    }
}

impl Mul for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Mat4) -> Mat4 {
        let mut out = Mat4::zero();
        for c in 0..4 {
            for r in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.m[k][r] * o.m[c][k];
                }
                out.m[c][r] = s;
            }
        }
        out
    }
}

pub fn quat_from_basis(cx: Vec3, cy: Vec3, cz: Vec3) -> Quat {
    let (m00, m11, m22) = (cx.x, cy.y, cz.z);
    let tr = m00 + m11 + m22;
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        Quat { w: 0.25 * s, x: (cy.z - cz.y) / s, y: (cz.x - cx.z) / s, z: (cx.y - cy.x) / s }
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat { w: (cy.z - cz.y) / s, x: 0.25 * s, y: (cy.x + cx.y) / s, z: (cz.x + cx.z) / s }
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat { w: (cz.x - cx.z) / s, x: (cy.x + cx.y) / s, y: 0.25 * s, z: (cz.y + cy.z) / s }
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat { w: (cx.y - cy.x) / s, x: (cz.x + cx.z) / s, y: (cz.y + cy.z) / s, z: 0.25 * s }
    };
    q.normalized()
}

/// Closest point parameter t in [0,1] on segment ab to p, and the distance.
pub fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> (f64, f64) {
    let ab = b - a;
    let l2 = ab.len2();
    let t = if l2 < 1e-18 { 0.0 } else { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) };
    (t, (p - (a + ab * t)).len())
}

/// Deterministic xorshift RNG for fixtures and sampling.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [0,1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trs_inverse_roundtrip() {
        let q = Quat::from_axis_angle(v3(0.3, 1.0, -0.2), 0.7);
        let m = Mat4::from_trs(v3(1.0, 2.0, 3.0), q, v3(1.0, 1.0, 1.0));
        let p = v3(0.5, -0.25, 2.0);
        let back = m.inverse().transform_point(m.transform_point(p));
        assert!((back - p).len() < 1e-12);
        let q2 = m.rotation();
        assert!(q2.dot(q).abs() > 1.0 - 1e-12);
        assert!((q.rotate(p) - m.transform_vector(p)).len() < 1e-12);
    }
}
