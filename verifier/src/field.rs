//! Goldilocks field  P = 2^64 − 2^32 + 1, bit-exact with protocol.py's int field.
//! Python uses arbitrary-precision ints; here u128 holds every product a·b
//! (a,b < P < 2^64 ⇒ a·b < 2^128) so the reduction is exact — no 32-bit limb
//! trick (that was only for numpy's lack of u128). Field elements are u64 in
//! [0, P).

pub const P: u64 = 0xFFFF_FFFF_0000_0001;        // 2^64 − 2^32 + 1
pub const GLOBAL_G: u64 = 7;                      // primitive root of F_P

#[inline]
pub fn add(a: u64, b: u64) -> u64 {
    let s = (a as u128 + b as u128) % P as u128;
    s as u64
}

#[inline]
pub fn sub(a: u64, b: u64) -> u64 {
    // (a − b) mod P, matching Python's % (always non-negative result).
    let pm = P as u128;
    (((a as u128 + pm) - b as u128) % pm) as u64
}

#[inline]
pub fn mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

/// Modular exponentiation a^e mod P (square-and-multiply) — matches Python pow.
pub fn pow(mut a: u64, mut e: u64) -> u64 {
    a %= P;
    let mut r: u64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, a);
        }
        a = mul(a, a);
        e >>= 1;
    }
    r
}

/// Fermat inverse a^(P−2) mod P (a ≠ 0) — matches protocol.inv.
#[inline]
pub fn inv(a: u64) -> u64 {
    pow(a % P, P - 2)
}
