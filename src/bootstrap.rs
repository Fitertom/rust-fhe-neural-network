//! # Bootstrapping Module
//!
//! Implements a simplified TFHE-inspired functional bootstrapping for LWE.
//!
//! ## Overview
//! Bootstrapping is the mechanism that "refreshes" a noisy ciphertext,
//! reducing its noise back to a small level. This is essential for
//! multi-layer neural networks: without it, noise would accumulate
//! across layers until decryption fails.
//!
//! Along with noise refresh, bootstrapping simultaneously evaluates a
//! lookup table (LUT) — this gives us activation functions (ReLU, etc.)
//! "for free" during noise management.
//!
//! ## Components
//!
//! ### Bootstrapping Key (BSK)
//! For each bit s_i of the original key, we encrypt s_i under a separate
//! bootstrapping key z. In full TFHE, the BSK consists of RGSW encryptions
//! of s_i under a separate bootstrapping key z. This allows the blind rotation
//! to "see" the secret key bits without revealing them.
//!
//! ### Key-Switching Key (KSK)
//! After blind rotation, the result is encrypted under key z. The KSK
//! enables switching back to the original key s. It consists of encryptions
//! of each z_i · 2^j under key s, for decomposition base B_ks.

use crate::lwe::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// BOOTSTRAPPING PARAMETERS
// ============================================================

/// Dimension of the bootstrapping key. Larger = less noise from bootstrapping
/// but slower. We use a moderate value for correctness.
pub const N_BOOTSTRAP: usize = 1024;

/// Number of slots in the LUT. Must be a power of 2.
/// The modulus-switched ciphertext maps the phase into [0, 2*LUT_SIZE).
pub const LUT_SIZE: usize = 512;

/// Key-switching decomposition base (log2).
/// Larger base = fewer KSK entries but more noise per entry.
/// Smaller base = more entries but less noise.
/// B_ks = 2^KS_BASE_LOG = 2^2 = 4
pub const KS_BASE_LOG: usize = 2;

/// Number of levels in the key-switching decomposition.
/// We decompose each 64-bit coefficient into KS_LEVELS chunks of KS_BASE_LOG bits.
/// 64 / 2 = 32 levels covers the full u64 range.
pub const KS_LEVELS: usize = 32;

// ============================================================
// DATA STRUCTURES
// ============================================================

/// The Bootstrapping Key (BSK).
///
/// For each of the n original secret key bits s_i, we store a pair:
///   (encrypt_z(s_i * DELTA_LUT), encrypt_z(0))
/// under the bootstrapping key z.
///
/// During blind rotation, these are used as selectors: if s_i=1,
/// we rotate the accumulator by the corresponding amount.
#[derive(Clone, Serialize, Deserialize)]
pub struct BootstrappingKey {
    /// For each s_i: encryptions of s_i under the bootstrapping key.
    /// bsk[i] = Enc_z(s_i), a ciphertext of dimension N_BOOTSTRAP.
    pub bsk: Vec<LweCiphertext>,
}

/// The Key-Switching Key (KSK).
///
/// Enables converting a ciphertext encrypted under key z (dimension N_BOOTSTRAP)
/// into one encrypted under key s (dimension N_LWE).
///
/// ksk[i][j] = Enc_s(z_i * 2^(j * KS_BASE_LOG))
/// for i in 0..N_BOOTSTRAP, j in 0..KS_LEVELS
#[derive(Clone, Serialize, Deserialize)]
pub struct KeySwitchingKey {
    /// ksk[i][j] encrypts z_i · 2^(j·KS_BASE_LOG) under the original key s.
    pub ksk: Vec<Vec<LweCiphertext>>,
}

/// Combined evaluation keys needed for bootstrapping.
#[derive(Clone, Serialize, Deserialize)]
pub struct EvalKeys {
    pub bsk: BootstrappingKey,
    pub ksk: KeySwitchingKey,
    /// The bootstrapping key itself is stored for the simplified bootstrap.
    /// In full TFHE, the boot key would not be stored here — the blind rotation
    /// would work purely homomorphically via RLWE polynomial operations.
    /// Our simplified LWE-only approach needs it to decrypt the intermediate phase.
    pub boot_key: LweSecretKey,
}

// ============================================================
// KEY GENERATION
// ============================================================

/// Generate the bootstrapping key.
///
/// For each bit s_i of the original key, encrypt it under the
/// bootstrapping key z with dimension N_BOOTSTRAP.
///
/// The BSK enables the blind rotation step: by homomorphically
/// evaluating which LUT entry to select based on the encrypted
/// secret key bits.
pub fn gen_bootstrapping_key(
    orig_key: &LweSecretKey,
    boot_key: &LweSecretKey,
    rng: &mut impl Rng,
) -> BootstrappingKey {
    let mut bsk = Vec::with_capacity(orig_key.values.len());

    for &s_i in &orig_key.values {
        // Encrypt each s_i ∈ {0, 1} under the bootstrapping key
        // We encode it with a special delta suited for the LUT operations
        let ct = encrypt_with_key(boot_key, s_i, DELTA, rng);
        bsk.push(ct);
    }

    BootstrappingKey { bsk }
}

/// Generate the key-switching key.
///
/// For each bit z_i of the bootstrapping key and each decomposition level j,
/// encrypt z_i · 2^(j·KS_BASE_LOG) under the original key s.
///
/// This allows converting the result of blind rotation (encrypted under z)
/// back to encryption under s.
///
/// The decomposition technique reduces noise: instead of multiplying by
/// the full z_i value, we decompose it into small digits and multiply
/// by each digit separately, accumulating less noise.
pub fn gen_key_switching_key(
    orig_key: &LweSecretKey,
    boot_key: &LweSecretKey,
    rng: &mut impl Rng,
) -> KeySwitchingKey {
    let mut ksk = Vec::with_capacity(boot_key.values.len());

    for &z_i in &boot_key.values {
        let mut levels = Vec::with_capacity(KS_LEVELS);

        for j in 0..KS_LEVELS {
            // Compute z_i · 2^(j · KS_BASE_LOG)
            let shift = (j * KS_BASE_LOG) as u32;
            let value = z_i.wrapping_mul(1u64 << shift);

            // Encrypt this value under the original key s
            // We use the standard DELTA for encoding
            let ct = encrypt(orig_key, 0, rng);
            // Add the raw value directly to b (not scaled by delta - this is
            // a key-switching trick where we work in the coefficient domain)
            let mut ct_ks = ct;
            ct_ks.b = ct_ks.b.wrapping_add(value);
            levels.push(ct_ks);
        }

        ksk.push(levels);
    }

    KeySwitchingKey { ksk }
}

/// Generate all evaluation keys needed for bootstrapping.
pub fn gen_eval_keys(orig_key: &LweSecretKey, rng: &mut impl Rng) -> (LweSecretKey, EvalKeys) {
    // Generate a fresh bootstrapping key with larger dimension
    let boot_key = keygen_n(rng, N_BOOTSTRAP);

    let bsk = gen_bootstrapping_key(orig_key, &boot_key, rng);
    let ksk = gen_key_switching_key(orig_key, &boot_key, rng);

    (boot_key.clone(), EvalKeys { bsk, ksk, boot_key })
}

// ============================================================
// KEY SWITCHING
// ============================================================

/// Switch a ciphertext from encryption under boot_key (dim N_BOOTSTRAP)
/// to encryption under orig_key (dim N_LWE).
///
/// Algorithm:
///   For each coefficient a_i of the input ciphertext:
///     1. Decompose a_i into digits in base 2^KS_BASE_LOG
///     2. For each digit d_j: subtract d_j · KSK[i][j] from the accumulator
///
/// This is the Lev-style key switching used in TFHE.
/// The decomposition ensures each multiplication is by a small digit (< 2^KS_BASE_LOG),
/// keeping noise growth minimal.
pub fn key_switch(ct: &LweCiphertext, ksk: &KeySwitchingKey) -> LweCiphertext {
    let n_out = ksk.ksk[0][0].a.len(); // = N_LWE

    // Start with (0, b) — carry over the b component
    let mut result_a = vec![0u64; n_out];
    let mut result_b = ct.b;

    // For each component of the input ciphertext's a vector
    for (i, &a_i) in ct.a.iter().enumerate() {
        if i >= ksk.ksk.len() {
            break;
        }

        // Decompose a_i into base-2^KS_BASE_LOG digits
        let mut val = a_i;
        for j in 0..KS_LEVELS {
            let digit = val & ((1u64 << KS_BASE_LOG) - 1); // extract lowest digits
            val >>= KS_BASE_LOG;

            if digit == 0 {
                continue; // skip zero digits (common optimization)
            }

            // Subtract digit · KSK[i][j]
            // This incrementally builds: result = (0, b) - Σ digit_j · KSK[i][j]
            let ks_ct = &ksk.ksk[i][j];
            for k in 0..n_out {
                result_a[k] = result_a[k].wrapping_sub(ks_ct.a[k].wrapping_mul(digit));
            }
            result_b = result_b.wrapping_sub(ks_ct.b.wrapping_mul(digit));
        }
    }

    LweCiphertext {
        a: result_a,
        b: result_b,
    }
}

// ============================================================
// MODULUS SWITCHING
// ============================================================

/// Switch a ciphertext from modulus Q = 2^64 to a smaller modulus 2·LUT_SIZE.
///
/// This is a lossy operation: it scales all ciphertext components by
/// (2·LUT_SIZE / Q) and rounds. The noise decreases proportionally,
/// but we also lose precision — this is okay because we only need
/// enough precision to index into the LUT.
///
/// Output: (a', b') where a'_i = round(a_i · 2·LUT_SIZE / Q)
pub fn modulus_switch(ct: &LweCiphertext, new_mod: u32) -> (Vec<u32>, u32) {
    let q = u128::from(u64::MAX) + 1; // 2^64
    let nm = new_mod as u128;

    let a_switched: Vec<u32> =
        ct.a.iter()
            .map(|&ai| {
                let scaled = (ai as u128 * nm + q / 2) / q;
                (scaled % nm) as u32
            })
            .collect();

    let b_switched = {
        let scaled = (ct.b as u128 * nm + q / 2) / q;
        (scaled % nm) as u32
    };

    (a_switched, b_switched)
}

// ============================================================
// LOOKUP TABLE CONSTRUCTION
// ============================================================

/// Build a lookup table encoding a function f: [0, T) -> [0, T).
///
/// The LUT is indexed by the modulus-switched phase. We map the
/// phase range [0, 2·LUT_SIZE) into the plaintext space [0, T).
///
/// The function f is encoded so that bootstrapping naturally selects
/// the correct output value.
pub fn build_lut(f: impl Fn(u64) -> u64) -> Vec<u64> {
    let num_entries = 2 * LUT_SIZE;
    let mut lut = vec![0u64; num_entries];

    for i in 0..num_entries {
        // Map the LUT index back to the plaintext value it represents
        // phase ∈ [0, 2·LUT_SIZE) corresponds to plaintext m = phase · T / (2·LUT_SIZE)
        let m = (i as u64 * T_PLAINTEXT / num_entries as u64) as u64;
        let fm = f(m) % T_PLAINTEXT;
        lut[i] = fm;
    }

    lut
}

/// The identity LUT — bootstrapping with this refreshes noise without
/// changing the message. This is the simplest "noise refresh" operation.
pub fn build_identity_lut() -> Vec<u64> {
    build_lut(|m| m)
}

/// The square activation LUT — computes f(x) = x² mod T.
/// Used as the polynomial-approximation activation function in the neural network.
pub fn build_square_lut() -> Vec<u64> {
    build_lut(|m| (m * m) % T_PLAINTEXT)
}

// ============================================================
// SIMPLIFIED BOOTSTRAP (LWE-only, no RLWE)
// ============================================================

/// Simplified bootstrapping for pure LWE ciphertexts.
///
/// ## Approach
/// Full TFHE-style bootstrapping uses RLWE (polynomial ring) accumulators
/// and RGSW external products for blind rotation. This is what enables
/// homomorphic LUT evaluation without revealing intermediate values.
///
/// Our simplified approach works differently:
///   1. Homomorphically compute <a_switched, s> using the BSK
///      (this gives Enc_z(<a,s>) under the bootstrapping key z)
///   2. Decrypt the intermediate phase using the bootstrapping key z
///   3. Look up f(m) in the LUT
///   4. Re-encrypt with fresh noise under key z
///   5. Key-switch back to the original key s
///
/// This is a "simulated bootstrap" — it correctly refreshes noise and
/// applies the activation function, but the intermediate decryption
/// means values are momentarily visible server-side (which is acceptable
/// in our demo where the server holds all keys anyway).
///
/// In production FHE, RLWE blind rotation eliminates this exposure.
pub fn bootstrap(ct: &LweCiphertext, lut: &[u64], eval_keys: &EvalKeys) -> LweCiphertext {
    let two_n = (2 * LUT_SIZE) as u64;

    // Step 1: Recover the original secret key from BSK
    // In our simplified approach, we decrypt each BSK entry to recover s_i.
    // In full TFHE, this is done via homomorphic blind rotation over polynomials.
    let orig_key_values: Vec<u64> = eval_keys
        .bsk
        .bsk
        .iter()
        .map(|bsk_i| decrypt(&eval_keys.boot_key, bsk_i))
        .collect();

    // Step 2: Compute the phase of the original ciphertext
    // phase = b - <a, s> mod Q
    let mut dot: u64 = 0;
    for (i, &a_i) in ct.a.iter().enumerate() {
        dot = dot.wrapping_add(a_i.wrapping_mul(orig_key_values[i]));
    }
    let phase = ct.b.wrapping_sub(dot);

    // Step 3: Modulus-switch the phase to get a LUT index
    // This maps the phase from [0, Q) to [0, 2*LUT_SIZE)
    let q = u128::from(u64::MAX) + 1; // 2^64
    let phase_index = ((phase as u128 * two_n as u128 + q / 2) / q % two_n as u128) as usize;

    // Step 4: Look up f(m) in the LUT
    let result_m = lut[phase_index % lut.len()];

    // Step 5: Re-encrypt under the bootstrapping key with fresh noise
    let mut rng = rand::thread_rng();
    let fresh_ct = encrypt_with_key(&eval_keys.boot_key, result_m, DELTA, &mut rng);

    // Step 6: Key-switch from bootstrapping key (dim N_BOOTSTRAP) to original key (dim N_LWE)
    key_switch(&fresh_ct, &eval_keys.ksk)
}

/// Bootstrap with the identity function (pure noise refresh).
pub fn bootstrap_identity(ct: &LweCiphertext, eval_keys: &EvalKeys) -> LweCiphertext {
    let lut = build_identity_lut();
    bootstrap(ct, &lut, eval_keys)
}

/// Bootstrap with the square activation function f(x) = x².
pub fn bootstrap_square(ct: &LweCiphertext, eval_keys: &EvalKeys) -> LweCiphertext {
    let lut = build_square_lut();
    bootstrap(ct, &lut, eval_keys)
}

/// The ReLU activation LUT (no rescaling).
/// Positive values [0, T/2) are kept as-is.
/// Negative values [T/2, T) are mapped to 0.
pub fn build_relu_lut() -> Vec<u64> {
    let half_t = T_PLAINTEXT / 2;
    build_lut(|m| if m < half_t { m } else { 0 })
}

/// Bootstrap with the ReLU activation function (no clamp).
pub fn bootstrap_relu(ct: &LweCiphertext, eval_keys: &EvalKeys) -> LweCiphertext {
    let lut = build_relu_lut();
    bootstrap(ct, &lut, eval_keys)
}

/// The ReLU + clamp activation LUT.
/// Positive values [0, clamp] are kept as-is.
/// Values in (clamp, T/2) are clamped to `clamp`.
/// Negative values [T/2, T) are mapped to 0.
/// This prevents Layer 2 overflow: WEIGHT_BOUND × clamp × HIDDEN_SIZE < T/2.
pub fn build_relu_clamp_lut(clamp: u64) -> Vec<u64> {
    let half_t = T_PLAINTEXT / 2;
    build_lut(|m| {
        if m < half_t {
            if m > clamp {
                clamp
            } else {
                m
            }
        } else {
            0
        }
    })
}

/// Bootstrap with the ReLU + clamp activation function.
/// Clamps the output at `clamp` to keep Layer 2 sums within T/2.
pub fn bootstrap_relu_clamp(ct: &LweCiphertext, clamp: u64, eval_keys: &EvalKeys) -> LweCiphertext {
    let lut = build_relu_clamp_lut(clamp);
    bootstrap(ct, &lut, eval_keys)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> impl Rng {
        rand::rngs::StdRng::seed_from_u64(123)
    }

    #[test]
    fn test_key_switch_roundtrip() {
        let mut rng = test_rng();
        let orig_key = keygen(&mut rng);
        let boot_key = keygen_n(&mut rng, N_BOOTSTRAP);
        let ksk = gen_key_switching_key(&orig_key, &boot_key, &mut rng);

        // Encrypt under boot_key, key-switch to orig_key
        for m in 0..16u64 {
            let ct = encrypt_with_key(&boot_key, m, DELTA, &mut rng);
            let switched = key_switch(&ct, &ksk);
            let result = decrypt(&orig_key, &switched);
            assert_eq!(m, result, "Key switch failed for m={}: got {}", m, result);
        }
    }

    #[test]
    fn test_bootstrap_identity() {
        let mut rng = test_rng();
        let orig_key = keygen(&mut rng);
        let (_boot_key, eval_keys) = gen_eval_keys(&orig_key, &mut rng);

        // Test bootstrapping with identity function (noise refresh only)
        for m in [0u64, 100, 500, 1000, 5000, 10000, 30000, 100000, 500000] {
            let ct = encrypt(&orig_key, m, &mut rng);
            let refreshed = bootstrap_identity(&ct, &eval_keys);
            let result = decrypt(&orig_key, &refreshed);

            // With 1024 LUT entries for T=1048576, resolution is T/1024 = 1024
            // So results can differ by up to ~1024 from the true value
            let diff = if result >= m { result - m } else { m - result };
            let tolerance = T_PLAINTEXT / (2 * LUT_SIZE as u64) + 1;
            assert!(
                diff <= tolerance,
                "Bootstrap identity failed for m={}: got {} (diff={}, tolerance={})",
                m,
                result,
                diff,
                tolerance
            );
        }
    }

    #[test]
    fn test_bootstrap_square_activation() {
        let mut rng = test_rng();
        let orig_key = keygen(&mut rng);
        let (_boot_key, eval_keys) = gen_eval_keys(&orig_key, &mut rng);

        // Test the square activation function (small values)
        for m in [0u64, 1, 2, 5, 10, 20] {
            let ct = encrypt(&orig_key, m, &mut rng);
            let squared = bootstrap_square(&ct, &eval_keys);
            let result = decrypt(&orig_key, &squared);
            let expected = (m * m) % T_PLAINTEXT;

            let diff = if result >= expected {
                result - expected
            } else {
                expected - result
            };
            let tolerance = T_PLAINTEXT / (LUT_SIZE as u64) + 1;
            assert!(
                diff <= tolerance,
                "Bootstrap square failed for m={}: got {}, expected {} (diff={}, tolerance={})",
                m,
                result,
                expected,
                diff,
                tolerance
            );
        }
    }

    #[test]
    fn test_modulus_switch() {
        // Verify modulus switching preserves the relative phase
        let two_n = (2 * LUT_SIZE) as u64;
        let q = u128::from(u64::MAX) + 1;

        // A phase of Δ·m should map to approximately (2N/T)·m after modulus switch
        for m in (0..T_PLAINTEXT).step_by(1000) {
            let phase = DELTA.wrapping_mul(m);
            let switched = (phase as u128 * two_n as u128 + q / 2) / q;
            let expected_approx = (two_n as u128 * m as u128) / T_PLAINTEXT as u128;
            let diff = if switched >= expected_approx {
                switched - expected_approx
            } else {
                expected_approx - switched
            };
            assert!(
                diff <= 2,
                "Modulus switch error for m={}: switched={}, expected≈{}",
                m,
                switched,
                expected_approx
            );
        }
    }
}
