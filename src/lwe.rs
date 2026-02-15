//! # LWE (Learning With Errors) Encryption Scheme
//!
//! A from-scratch implementation of the LWE symmetric encryption scheme using
//! only primitive types. No external cryptographic libraries.
//!
//! ## Parameters
//! - `N_LWE = 512`: Secret key dimension. Larger N → more secure, slower.
//! - `Q = 2^64`:    Ciphertext modulus. We use u64 wrapping arithmetic so
//!                  all additions/multiplications are implicitly mod 2^64.
//! - `T = 2^20`:    Plaintext modulus. Messages are integers in [0, T).
//!                  With 2^20 = 1,048,576 values, Layer 2 sums
//!                  (up to ~480K with clamp=1000 × weight-15 × 32 neurons)
//!                  fit in [−T/2, T/2) without modular overflow.
//! - `DELTA = Q/T`:  Scaling factor ≈ 2^44. This separates the message
//!                  bits (in the MSBs of the u64) from the noise (in the LSBs).
//! - `σ ≈ 2.0`:    Standard deviation of the noise distribution. We use a
//!                  centered binomial distribution with k=8.
//!
//! ## Noise Budget Analysis
//! After a matrix-vector product with 784 inputs and weights bounded by |w|≤15:
//!   σ_out ≈ σ · B · √784 ≈ 2.0 × 15 × 28 = 840
//!   6·σ_out = 5040 << Δ/2 = 2^43 = 8.8 × 10^12  ✓ (enormous margin)
//!
//! ## Encoding
//! A message m ∈ [0, T) is encoded in the MSBs of a u64:
//!   encoded = m * DELTA
//! Decoding recovers m by rounding:
//!   m = round(phase * T / Q) mod T
//! where phase = b - <a, s> (mod Q).

use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// LWE PARAMETERS — all chosen for educational correctness.
// In production FHE (e.g., TFHE), n ≥ 630, Q = 2^64, with
// carefully chosen noise for 128-bit security. Our parameters
// are smaller for performance in this demo.
// ============================================================

/// Secret key dimension. Each ciphertext contains a vector of this size.
pub const N_LWE: usize = 512;

/// Plaintext modulus. Messages are integers in [0, T_PLAINTEXT).
/// 20 bits gives us 1,048,576 distinct values — large enough that layer-2
/// dot products (up to ~480K with clamp=1000 × weight-15 × 32 neurons)
/// fit in [−T/2, T/2) without modular overflow.
pub const T_PLAINTEXT: u64 = 1 << 20; // = 1,048,576

/// Scaling factor: maps plaintext space [0, T) into the MSBs of u64.
/// DELTA = 2^64 / 2^20 = 2^44 = 17,592,186,044,416
/// The message occupies the top 20 bits; noise lives in the bottom 44 bits.
pub const DELTA: u64 = u64::MAX / T_PLAINTEXT + 1; // = 2^44

/// Number of terms to sum in the centered binomial distribution.
/// This gives σ ≈ √(k/2) ≈ √4 = 2.0 for k=8. We use k=8 for
/// a slightly wider distribution that better approximates a Gaussian.
const CBD_K: usize = 8;

// ============================================================
// DATA STRUCTURES
// ============================================================

/// LWE secret key: a vector of binary values {0, 1}^n.
/// Binary keys reduce noise growth during homomorphic operations
/// compared to uniform keys (since |s_i| ≤ 1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LweSecretKey {
    pub values: Vec<u64>,
}

/// LWE ciphertext: the pair (a, b) where:
///   a ∈ Z_Q^n  (random mask vector)
///   b ∈ Z_Q    (masked message + noise)
/// Invariant: b = <a, s> + Δ·m + e (mod Q)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LweCiphertext {
    /// Random mask vector, length N_LWE
    pub a: Vec<u64>,
    /// Scalar: <a,s> + Δ·m + e
    pub b: u64,
}

// ============================================================
// NOISE SAMPLING
// ============================================================

/// Sample from a centered binomial distribution CBD(k).
/// This is a standard technique in lattice-based crypto (used in Kyber,
/// NewHope, etc.) to efficiently approximate a discrete Gaussian.
///
/// Algorithm: sample 2k uniform bits, compute:
///   e = (Σ_{i=0}^{k-1} b_i) - (Σ_{i=k}^{2k-1} b_i)
///
/// Result is in [-k, k] with variance k/2.
/// For k=8: range [-8, 8], σ = √4 = 2.0
fn sample_noise(rng: &mut impl Rng) -> u64 {
    let mut sum_a: i32 = 0;
    let mut sum_b: i32 = 0;
    for _ in 0..CBD_K {
        sum_a += rng.gen_range(0..=1i32);
        sum_b += rng.gen_range(0..=1i32);
    }
    (sum_a - sum_b) as u64 // wraps for negative values
}

// ============================================================
// KEY GENERATION
// ============================================================

/// Generate a fresh LWE secret key.
/// Key entries are binary {0, 1} — this is a common choice that
/// reduces noise growth during scalar multiplication (since s_i ∈ {0,1},
/// multiplying by s_i doesn't amplify the error).
pub fn keygen(rng: &mut impl Rng) -> LweSecretKey {
    let values: Vec<u64> = (0..N_LWE).map(|_| rng.gen_range(0..=1u64)).collect();
    LweSecretKey { values }
}

/// Generate a secret key with a specific dimension (for bootstrapping keys).
pub fn keygen_n(rng: &mut impl Rng, n: usize) -> LweSecretKey {
    let values: Vec<u64> = (0..n).map(|_| rng.gen_range(0..=1u64)).collect();
    LweSecretKey { values }
}

// ============================================================
// ENCRYPTION / DECRYPTION
// ============================================================

/// Encrypt a plaintext message m ∈ [0, T_PLAINTEXT).
///
/// Ciphertext = (a, b) where:
///   a = random vector in Z_Q^n
///   b = <a, s> + Δ·m + e (mod Q)
///
/// The message is placed in the MSBs via multiplication by Δ.
/// The noise e is small (|e| << Δ/2), so it doesn't affect the
/// MSBs during decryption.
pub fn encrypt(key: &LweSecretKey, m: u64, rng: &mut impl Rng) -> LweCiphertext {
    debug_assert!(
        m < T_PLAINTEXT,
        "Message {} must be < T_PLAINTEXT={}",
        m,
        T_PLAINTEXT
    );

    let n = key.values.len();

    // Random mask vector a
    let a: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

    // Inner product <a, s> mod Q (wrapping arithmetic)
    let mut dot: u64 = 0;
    for i in 0..n {
        dot = dot.wrapping_add(a[i].wrapping_mul(key.values[i]));
    }

    // b = <a,s> + Δ·m + e
    let noise = sample_noise(rng);
    let b = dot.wrapping_add(DELTA.wrapping_mul(m)).wrapping_add(noise);

    LweCiphertext { a, b }
}

/// Encrypt with a specific-dimension key (for bootstrapping).
pub fn encrypt_with_key(
    key: &LweSecretKey,
    m: u64,
    delta: u64,
    rng: &mut impl Rng,
) -> LweCiphertext {
    let n = key.values.len();
    let a: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
    let mut dot: u64 = 0;
    for i in 0..n {
        dot = dot.wrapping_add(a[i].wrapping_mul(key.values[i]));
    }
    let noise = sample_noise(rng);
    let b = dot.wrapping_add(delta.wrapping_mul(m)).wrapping_add(noise);
    LweCiphertext { a, b }
}

/// Decrypt a ciphertext to recover the plaintext message.
///
/// Algorithm:
///   1. Compute phase = b - <a, s> (mod Q)
///   2. This gives Δ·m + e (mod Q)
///   3. Round: m = round(phase · T / Q) mod T
///
/// The rounding works because:
///   phase = Δ·m + e = (Q/T)·m + e
///   phase · T / Q = m + e·T/Q ≈ m  (since |e| << Q/T)
pub fn decrypt(key: &LweSecretKey, ct: &LweCiphertext) -> u64 {
    let n = key.values.len();

    // Compute phase = b - <a, s> (mod Q)
    let mut dot: u64 = 0;
    for i in 0..n {
        dot = dot.wrapping_add(ct.a[i].wrapping_mul(key.values[i]));
    }
    let phase = ct.b.wrapping_sub(dot);

    // Round to nearest multiple of Δ:
    // m = round(phase * T / Q)
    // We add Δ/2 before integer division to achieve rounding.
    let shifted = phase.wrapping_add(DELTA / 2);
    let m = shifted / DELTA;
    m % T_PLAINTEXT
}

/// Create a "trivial" ciphertext encrypting m with zero noise.
/// a = 0, b = Δ·m. This requires no secret key — anyone can create it.
/// Useful for encoding known constants (e.g., bias terms) into ciphertext
/// form for homomorphic operations.
pub fn trivial_encrypt(m: u64) -> LweCiphertext {
    LweCiphertext {
        a: vec![0u64; N_LWE],
        b: DELTA.wrapping_mul(m),
    }
}

/// Create a trivial ciphertext with a given dimension.
pub fn trivial_encrypt_n(m: u64, n: usize) -> LweCiphertext {
    LweCiphertext {
        a: vec![0u64; n],
        b: DELTA.wrapping_mul(m),
    }
}

// ============================================================
// HOMOMORPHIC OPERATIONS
// ============================================================

/// Homomorphic addition of two ciphertexts.
///
/// If ct1 encrypts m1 and ct2 encrypts m2, the result encrypts m1 + m2.
///
/// Proof:
///   ct1 = (a1, <a1,s> + Δ·m1 + e1)
///   ct2 = (a2, <a2,s> + Δ·m2 + e2)
///   ct1 + ct2 = (a1+a2, <a1+a2, s> + Δ·(m1+m2) + (e1+e2))
///
/// Note: noise ADDS. After many additions, noise may overflow Δ/2.
pub fn homo_add(ct1: &LweCiphertext, ct2: &LweCiphertext) -> LweCiphertext {
    debug_assert_eq!(ct1.a.len(), ct2.a.len(), "Ciphertext dimensions must match");
    let a: Vec<u64> = ct1
        .a
        .iter()
        .zip(ct2.a.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    let b = ct1.b.wrapping_add(ct2.b);
    LweCiphertext { a, b }
}

/// Homomorphic subtraction of two ciphertexts.
/// Result encrypts m1 - m2 (mod T_PLAINTEXT).
pub fn homo_sub(ct1: &LweCiphertext, ct2: &LweCiphertext) -> LweCiphertext {
    debug_assert_eq!(ct1.a.len(), ct2.a.len());
    let a: Vec<u64> = ct1
        .a
        .iter()
        .zip(ct2.a.iter())
        .map(|(&x, &y)| x.wrapping_sub(y))
        .collect();
    let b = ct1.b.wrapping_sub(ct2.b);
    LweCiphertext { a, b }
}

/// Scalar multiplication of a ciphertext by a plaintext constant.
///
/// If ct encrypts m, the result encrypts c·m (mod T_PLAINTEXT).
///
/// Proof:
///   ct = (a, <a,s> + Δ·m + e)
///   c·ct = (c·a, c·<a,s> + c·Δ·m + c·e)
///        = (c·a, <c·a, s> + Δ·(c·m) + c·e)
///
/// CRITICAL: noise is multiplied by |c|. Large scalars quickly
/// exhaust the noise budget. This is why we quantize NN weights
/// to small values (|w| ≤ 15).
pub fn scalar_mul(ct: &LweCiphertext, c: i32) -> LweCiphertext {
    // Convert signed scalar to u64 for wrapping arithmetic
    let c_u64 = c as i64 as u64;
    let a: Vec<u64> = ct.a.iter().map(|&x| x.wrapping_mul(c_u64)).collect();
    let b = ct.b.wrapping_mul(c_u64);
    LweCiphertext { a, b }
}

/// Negate a ciphertext: encrypts -m (mod T_PLAINTEXT).
pub fn negate(ct: &LweCiphertext) -> LweCiphertext {
    let a: Vec<u64> = ct.a.iter().map(|&x| (!x).wrapping_add(1)).collect();
    let b = (!ct.b).wrapping_add(1);
    LweCiphertext { a, b }
}

// ============================================================
// UTILITY: compute the "phase" (for debugging / analysis)
// ============================================================

/// Compute the raw phase b - <a,s> without rounding.
/// This reveals Δ·m + e — useful for inspecting noise levels.
pub fn compute_phase(key: &LweSecretKey, ct: &LweCiphertext) -> u64 {
    let mut dot: u64 = 0;
    for i in 0..key.values.len() {
        dot = dot.wrapping_add(ct.a[i].wrapping_mul(key.values[i]));
    }
    ct.b.wrapping_sub(dot)
}

/// Estimate the noise magnitude in a ciphertext.
/// Returns |e| = |phase - Δ·m|, which tells us how far the noise
/// has grown from the ideal encoding.
pub fn estimate_noise(key: &LweSecretKey, ct: &LweCiphertext, expected_m: u64) -> u64 {
    let phase = compute_phase(key, ct);
    let ideal = DELTA.wrapping_mul(expected_m);
    let diff = phase.wrapping_sub(ideal);
    // Return the smaller of diff and Q-diff (since we're mod Q)
    std::cmp::min(diff, (!diff).wrapping_add(1))
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> impl Rng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let mut rng = test_rng();
        let key = keygen(&mut rng);

        // Test a range of values in plaintext space (T is too large to test all)
        for m in (0..T_PLAINTEXT).step_by(1000) {
            let ct = encrypt(&key, m, &mut rng);
            let decrypted = decrypt(&key, &ct);
            assert_eq!(
                m, decrypted,
                "Failed roundtrip for m={}: got {}",
                m, decrypted
            );
        }
    }

    #[test]
    fn test_homomorphic_addition() {
        let mut rng = test_rng();
        let key = keygen(&mut rng);

        for _ in 0..100 {
            let a = rng.gen_range(0..T_PLAINTEXT / 4);
            let b = rng.gen_range(0..T_PLAINTEXT / 4);

            let ct_a = encrypt(&key, a, &mut rng);
            let ct_b = encrypt(&key, b, &mut rng);
            let ct_sum = homo_add(&ct_a, &ct_b);
            let result = decrypt(&key, &ct_sum);

            assert_eq!(
                (a + b) % T_PLAINTEXT,
                result,
                "Homo add failed: {} + {} = {} (expected {})",
                a,
                b,
                result,
                (a + b) % T_PLAINTEXT
            );
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        let mut rng = test_rng();
        let key = keygen(&mut rng);

        for _ in 0..100 {
            let m = rng.gen_range(0..T_PLAINTEXT / 4);
            let c = rng.gen_range(-15..=15i32);

            let ct = encrypt(&key, m, &mut rng);
            let ct_scaled = scalar_mul(&ct, c);
            let result = decrypt(&key, &ct_scaled);

            let expected = ((m as i64 * c as i64) % T_PLAINTEXT as i64 + T_PLAINTEXT as i64)
                % T_PLAINTEXT as i64;
            assert_eq!(
                expected as u64, result,
                "Scalar mul failed: {} * {} = {} (expected {})",
                c, m, result, expected
            );
        }
    }

    #[test]
    fn test_trivial_encrypt() {
        let mut rng = test_rng();
        let key = keygen(&mut rng);

        for m in (0..T_PLAINTEXT).step_by(1000) {
            let ct = trivial_encrypt(m);
            let result = decrypt(&key, &ct);
            assert_eq!(m, result, "Trivial encrypt failed for m={}", m);
        }
    }

    #[test]
    fn test_noise_growth_under_additions() {
        let mut rng = test_rng();
        let key = keygen(&mut rng);

        // Sum 784 ciphertexts (simulating a dot product with weights=1)
        let mut acc = trivial_encrypt(0);
        let mut expected_sum: u64 = 0;
        for _ in 0..784 {
            let m = 1u64;
            expected_sum = (expected_sum + m) % T_PLAINTEXT;
            let ct = encrypt(&key, m, &mut rng);
            acc = homo_add(&acc, &ct);
        }
        let result = decrypt(&key, &acc);
        assert_eq!(
            expected_sum, result,
            "After 784 additions: expected {}, got {}",
            expected_sum, result
        );

        // Check noise level is still well within budget
        let noise = estimate_noise(&key, &acc, expected_sum);
        assert!(
            noise < DELTA / 4,
            "Noise {} exceeds Δ/4 = {} after 784 additions",
            noise,
            DELTA / 4
        );
    }
}
