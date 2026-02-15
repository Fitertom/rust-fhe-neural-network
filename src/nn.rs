//! # Neural Network Module
//!
//! A simple feed-forward neural network for digit recognition.
//! Architecture: Input(784) → Hidden(32) → ReLU+Clamp → Output(10)
//!
//! ## Training
//! Training happens in plaintext floating-point:
//!   1. Forward pass with ReLU (for stable gradient flow during training)
//!   2. Backward pass with standard backpropagation
//!   3. After training, weights are quantized to integers in [-B, B]
//!      where B=15, for compatibility with FHE inference.
//!
//! ## Quantized Inference
//! The quantized network uses integer arithmetic:
//!   - Weights: i32 in [-15, 15]
//!   - Activations: u64 in [0, T_PLAINTEXT)
//!   - ReLU+clamp activation is applied between layers
//!
//! This integer network is what gets evaluated homomorphically.

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Input dimension: 28×28 = 784 pixels
pub const INPUT_SIZE: usize = 784;
/// Hidden layer size — reduced from 64 to keep layer-2 sums within T/2
pub const HIDDEN_SIZE: usize = 32;
/// Output classes: digits 0-9
pub const OUTPUT_SIZE: usize = 10;
/// Weight quantization bound: weights are clamped to [-WEIGHT_BOUND, WEIGHT_BOUND]
pub const WEIGHT_BOUND: f64 = 15.0;

/// Input pixel quantization scale: float [0,1] → integer [0, INPUT_SCALE]
/// 4-bit precision (16 levels). With u64 LWE we have room for more precision.
/// Constraint: WEIGHT_BOUND × INPUT_SCALE × INPUT_SIZE < T/2
pub const INPUT_SCALE: u64 = 15;

/// Hidden activation scale after bootstrap ReLU+rescale LUT.
/// The bootstrap LUT maps ReLU output from [0, T/2) to [0, HIDDEN_SCALE].
/// Constraint: WEIGHT_BOUND × HIDDEN_SCALE × HIDDEN_SIZE < T/2
/// 15 × 1000 × 32 = 480,000 < 524,288 = T/2 ✓
pub const HIDDEN_SCALE: u64 = 1000;

/// Plaintext modulus for quantized inference
pub const T_MOD: u64 = crate::lwe::T_PLAINTEXT;

// ============================================================
// NETWORK STRUCTURE
// ============================================================

/// Floating-point network used during training.
#[derive(Clone)]
pub struct NetworkF64 {
    /// Layer 1 weights: HIDDEN_SIZE × INPUT_SIZE
    pub w1: Vec<Vec<f64>>,
    /// Layer 1 biases: HIDDEN_SIZE
    pub b1: Vec<f64>,
    /// Layer 2 weights: OUTPUT_SIZE × HIDDEN_SIZE
    pub w2: Vec<Vec<f64>>,
    /// Layer 2 biases: OUTPUT_SIZE
    pub b2: Vec<f64>,
    /// When true, forward pass applies fake quantization (QAT mode).
    pub qat_enabled: bool,
}

/// Quantized integer network used for FHE inference.
/// Weights are small integers in [-WEIGHT_BOUND, WEIGHT_BOUND].
#[derive(Clone, Serialize, Deserialize)]
pub struct NetworkQuantized {
    /// Layer 1 weights (i32 for scalar multiplication compatibility)
    pub w1: Vec<Vec<i32>>,
    /// Layer 1 biases (u64 in [0, T_PLAINTEXT))
    pub b1: Vec<u64>,
    /// Layer 2 weights
    pub w2: Vec<Vec<i32>>,
    /// Layer 2 biases
    pub b2: Vec<u64>,
}

// ============================================================
// FLOATING-POINT TRAINING
// ============================================================

impl NetworkF64 {
    /// Initialize with small random weights (Xavier initialization).
    pub fn new(rng: &mut impl Rng) -> Self {
        let w1_scale = (2.0 / INPUT_SIZE as f64).sqrt();
        let w2_scale = (2.0 / HIDDEN_SIZE as f64).sqrt();

        let w1: Vec<Vec<f64>> = (0..HIDDEN_SIZE)
            .map(|_| {
                (0..INPUT_SIZE)
                    .map(|_| rng.gen_range(-1.0..1.0) * w1_scale)
                    .collect()
            })
            .collect();

        let b1 = vec![0.0; HIDDEN_SIZE];

        let w2: Vec<Vec<f64>> = (0..OUTPUT_SIZE)
            .map(|_| {
                (0..HIDDEN_SIZE)
                    .map(|_| rng.gen_range(-1.0..1.0) * w2_scale)
                    .collect()
            })
            .collect();

        let b2 = vec![0.0; OUTPUT_SIZE];

        NetworkF64 {
            w1,
            b1,
            w2,
            b2,
            qat_enabled: false,
        }
    }

    /// Compute per-layer scale factor: WEIGHT_BOUND / max(|weights|)
    fn layer_scale(weights: &[Vec<f64>]) -> f64 {
        let max_w = weights
            .iter()
            .flatten()
            .map(|w| w.abs())
            .fold(0.0f64, f64::max)
            .max(1e-8);
        WEIGHT_BOUND / max_w
    }

    /// Fake-quantize a weight: round to integer grid, clamp, convert back.
    /// STE: in backward pass, gradient flows through as if this was identity.
    #[inline]
    fn fake_quantize(w: f64, scale: f64) -> f64 {
        let q = (w * scale).round().clamp(-WEIGHT_BOUND, WEIGHT_BOUND);
        q / scale // back to float space
    }

    /// Forward pass through the network.
    /// Returns (hidden activations, output logits) for use in backprop.
    ///
    /// Uses ReLU + dynamic clamp to match the bootstrap ReLU+clamp LUT.
    /// In QAT mode, clamp = HIDDEN_SCALE / (scale1 * INPUT_SCALE) to prevent
    /// Layer 2 overflow in FHE (where values are mod T_PLAINTEXT).
    ///
    /// QAT simulates:
    /// - Input quantization: float → round to INPUT_SCALE grid → back to float
    /// - Weight fake-quantization: round to integer grid, clamp, convert back
    /// - Bias fake-quantization: same scale as weights
    pub fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // clamp removed

        // Simulate input quantization: [0,1] → [0, INPUT_SCALE] → round → back
        let qinput: Vec<f64> = if self.qat_enabled {
            input
                .iter()
                .map(|&x| {
                    let q = (x * INPUT_SCALE as f64)
                        .round()
                        .clamp(0.0, INPUT_SCALE as f64);
                    q / INPUT_SCALE as f64
                })
                .collect()
        } else {
            input.to_vec()
        };

        // Per-layer scales for QAT
        let scale1 = if self.qat_enabled {
            Self::layer_scale(&self.w1)
        } else {
            0.0
        };
        let scale2 = if self.qat_enabled {
            Self::layer_scale(&self.w2)
        } else {
            0.0
        };

        // Layer 1: z1 = W1 · input + b1
        let mut z1 = vec![0.0; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            let b = if self.qat_enabled {
                let qb = (self.b1[j] * scale1).round() / scale1;
                qb
            } else {
                self.b1[j]
            };
            let mut sum = b;
            for i in 0..INPUT_SIZE {
                let w = if self.qat_enabled {
                    Self::fake_quantize(self.w1[j][i], scale1)
                } else {
                    self.w1[j][i]
                };
                sum += w * qinput[i];
            }
            z1[j] = sum;
        }

        // ReLU + clamp: match the bootstrap_relu_clamp LUT behavior.
        // In QAT mode, clamp at float equivalent of HIDDEN_SCALE to prevent
        // Layer 2 overflow: WEIGHT_BOUND × HIDDEN_SCALE × HIDDEN_SIZE < T/2.
        let h1: Vec<f64> = if self.qat_enabled {
            let clamp_f = HIDDEN_SCALE as f64 / (scale1 * INPUT_SCALE as f64);
            z1.iter().map(|&x| x.max(0.0).min(clamp_f)).collect()
        } else {
            z1.iter().map(|&x| x.max(0.0)).collect()
        };

        // Layer 2: z2 = W2 · h1 + b2
        let mut z2 = vec![0.0; OUTPUT_SIZE];
        for j in 0..OUTPUT_SIZE {
            let b = if self.qat_enabled {
                let qb = (self.b2[j] * scale2).round() / scale2;
                qb
            } else {
                self.b2[j]
            };
            let mut sum = b;
            for i in 0..HIDDEN_SIZE {
                let w = if self.qat_enabled {
                    Self::fake_quantize(self.w2[j][i], scale2)
                } else {
                    self.w2[j][i]
                };
                sum += w * h1[i];
            }
            z2[j] = sum;
        }

        (h1, z2)
    }

    /// Softmax + cross-entropy loss backward pass.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Train on a batch of (input, label) pairs.
    /// Uses SGD with learning rate lr.
    /// Gradient computation is parallelized across samples with Rayon.
    pub fn train_step(&mut self, batch: &[(Vec<f64>, u8)], lr: f64) {
        let batch_size = batch.len() as f64;

        // Per-sample gradient struct for parallel reduction
        struct Grads {
            dw1: Vec<Vec<f64>>,
            db1: Vec<f64>,
            dw2: Vec<Vec<f64>>,
            db2: Vec<f64>,
        }

        impl Grads {
            fn zero() -> Self {
                Grads {
                    dw1: vec![vec![0.0; INPUT_SIZE]; HIDDEN_SIZE],
                    db1: vec![0.0; HIDDEN_SIZE],
                    dw2: vec![vec![0.0; HIDDEN_SIZE]; OUTPUT_SIZE],
                    db2: vec![0.0; OUTPUT_SIZE],
                }
            }

            fn merge(mut self, other: Self) -> Self {
                for j in 0..HIDDEN_SIZE {
                    self.db1[j] += other.db1[j];
                    for i in 0..INPUT_SIZE {
                        self.dw1[j][i] += other.dw1[j][i];
                    }
                }
                for j in 0..OUTPUT_SIZE {
                    self.db2[j] += other.db2[j];
                    for i in 0..HIDDEN_SIZE {
                        self.dw2[j][i] += other.dw2[j][i];
                    }
                }
                self
            }
        }

        // Snapshot weights for parallel forward/backward passes
        let w1 = &self.w1;
        let b1 = &self.b1;
        let w2 = &self.w2;
        let b2 = &self.b2;
        let qat = self.qat_enabled;

        // Per-layer scale factors for QAT fake quantization
        let scale1 = if qat { Self::layer_scale(w1) } else { 0.0 };
        let scale2 = if qat { Self::layer_scale(w2) } else { 0.0 };

        // Parallel gradient computation: each sample computes its own gradients
        let grads = batch
            .par_iter()
            .fold(Grads::zero, |mut acc, (input, label)| {
                // Forward pass (using shared weights, with optional fake quantization)
                let mut z1 = vec![0.0; HIDDEN_SIZE];
                for j in 0..HIDDEN_SIZE {
                    let mut sum = b1[j];
                    for i in 0..INPUT_SIZE {
                        let w = if qat {
                            Self::fake_quantize(w1[j][i], scale1)
                        } else {
                            w1[j][i]
                        };
                        sum += w * input[i];
                    }
                    z1[j] = sum;
                }
                // ReLU + dynamic clamp matching FHE bootstrap LUT
                let clamp = if qat {
                    HIDDEN_SCALE as f64 / (scale1 * INPUT_SCALE as f64)
                } else {
                    f64::INFINITY
                };
                let h1: Vec<f64> = z1.iter().map(|&x| x.max(0.0).min(clamp)).collect();

                let mut z2 = vec![0.0; OUTPUT_SIZE];
                for j in 0..OUTPUT_SIZE {
                    let mut sum = b2[j];
                    for i in 0..HIDDEN_SIZE {
                        let w = if qat {
                            Self::fake_quantize(w2[j][i], scale2)
                        } else {
                            w2[j][i]
                        };
                        sum += w * h1[i];
                    }
                    z2[j] = sum;
                }

                // Softmax
                let max_val = z2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = z2.iter().map(|&x| (x - max_val).exp()).collect();
                let exp_sum: f64 = exps.iter().sum();
                let mut dz2: Vec<f64> = exps.iter().map(|&e| e / exp_sum).collect();
                dz2[*label as usize] -= 1.0;

                // Gradients for W2, b2
                for j in 0..OUTPUT_SIZE {
                    acc.db2[j] += dz2[j];
                    for i in 0..HIDDEN_SIZE {
                        acc.dw2[j][i] += dz2[j] * h1[i];
                    }
                }

                // Backprop through Layer 2
                let mut dh1 = vec![0.0; HIDDEN_SIZE];
                for i in 0..HIDDEN_SIZE {
                    for j in 0..OUTPUT_SIZE {
                        dh1[i] += w2[j][i] * dz2[j];
                    }
                }

                // Backprop through clamped ReLU: gradient is 1 when 0 < x < clamp, else 0
                let dz1: Vec<f64> = dh1
                    .iter()
                    .zip(z1.iter())
                    .map(|(&dh, &z)| if z > 0.0 && z < clamp { dh } else { 0.0 })
                    .collect();

                // Gradients for W1, b1
                for j in 0..HIDDEN_SIZE {
                    acc.db1[j] += dz1[j];
                    for i in 0..INPUT_SIZE {
                        acc.dw1[j][i] += dz1[j] * input[i];
                    }
                }

                acc
            })
            .reduce(Grads::zero, |a, b| a.merge(b));

        // SGD update: W -= lr * dW / batch_size
        for j in 0..HIDDEN_SIZE {
            self.b1[j] -= lr * grads.db1[j] / batch_size;
            for i in 0..INPUT_SIZE {
                self.w1[j][i] -= lr * grads.dw1[j][i] / batch_size;
            }
        }
        for j in 0..OUTPUT_SIZE {
            self.b2[j] -= lr * grads.db2[j] / batch_size;
            for i in 0..HIDDEN_SIZE {
                self.w2[j][i] -= lr * grads.dw2[j][i] / batch_size;
            }
        }
    }

    /// Evaluate accuracy on a dataset (parallelized).
    pub fn accuracy(&self, data: &[(Vec<f64>, u8)]) -> f64 {
        let correct: usize = data
            .par_iter()
            .filter(|(input, label)| {
                let (_, logits) = self.forward(input);
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                pred == *label as usize
            })
            .count();
        correct as f64 / data.len() as f64
    }

    /// Full training loop with QAT, optional validation, and early stopping.
    ///
    /// If `test_data` is provided, tracks test accuracy and keeps the best model.
    /// Early stopping: if test accuracy doesn't improve for `patience` epochs, stop.
    pub fn train(
        &mut self,
        train_data: &[(Vec<f64>, u8)],
        test_data: Option<&[(Vec<f64>, u8)]>,
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) {
        let mut rng = rand::thread_rng();
        self.qat_enabled = true;
        println!("  QAT enabled from epoch 1");

        let patience = 50;
        let mut best_test_acc = 0.0f64;
        let mut no_improve = 0usize;
        let mut best_w1 = self.w1.clone();
        let mut best_b1 = self.b1.clone();
        let mut best_w2 = self.w2.clone();
        let mut best_b2 = self.b2.clone();

        for epoch in 0..epochs {
            let progress = epoch as f64 / epochs as f64;
            let current_lr =
                lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));

            let mut indices: Vec<usize> = (0..train_data.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for chunk in indices.chunks(batch_size) {
                let batch: Vec<_> = chunk.iter().map(|&i| train_data[i].clone()).collect();
                self.train_step(&batch, current_lr);
            }

            let train_acc = self.accuracy(train_data);

            if let Some(test) = test_data {
                let test_acc = self.accuracy(test);
                let star = if test_acc > best_test_acc { " ★" } else { "" };
                println!(
                    "  Epoch {}/{}: train={:.1}% test={:.1}% (lr={:.5}) [QAT]{}",
                    epoch + 1,
                    epochs,
                    train_acc * 100.0,
                    test_acc * 100.0,
                    current_lr,
                    star
                );

                if test_acc > best_test_acc {
                    best_test_acc = test_acc;
                    no_improve = 0;
                    best_w1 = self.w1.clone();
                    best_b1 = self.b1.clone();
                    best_w2 = self.w2.clone();
                    best_b2 = self.b2.clone();
                } else {
                    no_improve += 1;
                    if no_improve >= patience {
                        println!(
                            "  ✓ Early stop: no improvement for {} epochs. Best test={:.1}%",
                            patience,
                            best_test_acc * 100.0
                        );
                        break;
                    }
                }
            } else {
                println!(
                    "  Epoch {}/{}: accuracy = {:.1}% (lr={:.5}) [QAT]",
                    epoch + 1,
                    epochs,
                    train_acc * 100.0,
                    current_lr
                );
                if train_acc >= 0.92 {
                    println!("  ✓ Reached {:.1}% — stopping early.", train_acc * 100.0);
                    break;
                }
            }
        }

        if test_data.is_some() {
            println!(
                "  Restoring best model (test={:.1}%)",
                best_test_acc * 100.0
            );
            self.w1 = best_w1;
            self.b1 = best_b1;
            self.w2 = best_w2;
            self.b2 = best_b2;
        }
    }

    /// Quantize the trained network to integer weights.
    ///
    /// Procedure:
    ///   1. Per-layer scaling: scale each layer's weights independently
    ///   2. Round to nearest integer
    ///   3. Clamp to [-WEIGHT_BOUND, WEIGHT_BOUND]
    ///   4. Map biases into [0, T_PLAINTEXT) space
    pub fn quantize(&self) -> NetworkQuantized {
        // Per-layer scaling: each layer gets its own scale factor
        // This prevents one layer's large weights from crushing the other layer
        let max_w1 = self
            .w1
            .iter()
            .flatten()
            .map(|w| w.abs())
            .fold(0.0f64, f64::max)
            .max(1e-8);
        let max_w2 = self
            .w2
            .iter()
            .flatten()
            .map(|w| w.abs())
            .fold(0.0f64, f64::max)
            .max(1e-8);

        let scale1 = WEIGHT_BOUND / max_w1;
        let scale2 = WEIGHT_BOUND / max_w2;

        let w1: Vec<Vec<i32>> = self
            .w1
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&w| (w * scale1).round().clamp(-WEIGHT_BOUND, WEIGHT_BOUND) as i32)
                    .collect()
            })
            .collect();

        let w2: Vec<Vec<i32>> = self
            .w2
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&w| (w * scale2).round().clamp(-WEIGHT_BOUND, WEIGHT_BOUND) as i32)
                    .collect()
            })
            .collect();

        // Quantize biases: each layer uses its own scale
        let b1: Vec<u64> = self
            .b1
            .iter()
            .map(|&b| {
                let scaled = (b * scale1).round() as i64;
                ((scaled % T_MOD as i64 + T_MOD as i64) % T_MOD as i64) as u64
            })
            .collect();

        let b2: Vec<u64> = self
            .b2
            .iter()
            .map(|&b| {
                let scaled = (b * scale2).round() as i64;
                ((scaled % T_MOD as i64 + T_MOD as i64) % T_MOD as i64) as u64
            })
            .collect();

        NetworkQuantized { w1, b1, w2, b2 }
    }
}

// ============================================================
// QUANTIZED INFERENCE (Plaintext Integer)
// ============================================================

impl NetworkQuantized {
    /// Run plaintext integer inference — mirrors FHE exactly.
    ///
    /// Uses modular arithmetic matching the encrypted path:
    /// 1. Layer 1: dot product mod T (sums fit in [-T/2, T/2) by design)
    /// 2. ReLU + rescale: same LUT as bootstrap — values in [0, T/2) kept
    ///    and rescaled to [0, HIDDEN_SCALE], values in [T/2, T) zeroed
    /// 3. Layer 2: dot product, output interpreted as signed mod T for argmax
    pub fn infer_plaintext(&self, input: &[u64]) -> Vec<u64> {
        assert_eq!(input.len(), INPUT_SIZE);
        let t = T_MOD as i64;
        let half_t = t / 2;

        // Layer 1: z1[j] = Σ_i W1[j][i] · input[i] + b1[j]
        let mut z1 = vec![0i64; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            // Interpret bias as signed: values >= T/2 are negative
            let bias = if self.b1[j] as i64 >= half_t {
                self.b1[j] as i64 - t
            } else {
                self.b1[j] as i64
            };
            let mut sum = bias;
            for i in 0..INPUT_SIZE {
                sum += self.w1[j][i] as i64 * input[i] as i64;
            }
            z1[j] = sum;
        }

        // ReLU activation (same as bootstrap LUT):
        // Reduce mod T → signed interpretation → ReLU (keep positive, zero negative)
        let h1: Vec<u64> = z1
            .iter()
            .map(|&x| {
                let m = ((x % t) + t) % t; // reduce to [0, T)
                if m < half_t {
                    std::cmp::min(m as u64, HIDDEN_SCALE) // ReLU + clamp
                } else {
                    0 // Negative: zeroed by ReLU
                }
            })
            .collect();

        // Layer 2: z2[j] = Σ_i W2[j][i] · h1[i] + b2[j]
        // Output interpreted as signed for argmax
        let mut output = vec![0u64; OUTPUT_SIZE];
        for j in 0..OUTPUT_SIZE {
            // Interpret bias as signed: values >= T/2 are negative
            let bias2 = if self.b2[j] as i64 >= half_t {
                self.b2[j] as i64 - t
            } else {
                self.b2[j] as i64
            };
            let mut sum = bias2;
            for i in 0..HIDDEN_SIZE {
                sum += self.w2[j][i] as i64 * h1[i] as i64;
            }
            // Reduce mod T for FHE compatibility
            output[j] = ((sum % t + t) % t) as u64;
        }

        output
    }

    /// Predict the digit (argmax of output with signed mod-T interpretation).
    pub fn predict_plaintext(&self, input: &[u64]) -> usize {
        let output = self.infer_plaintext(input);
        let half_t = T_MOD / 2;
        // Interpret mod-T values as signed: [0, T/2) positive, [T/2, T) negative
        output
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| {
                if v < half_t {
                    v as i64
                } else {
                    v as i64 - T_MOD as i64
                }
            })
            .unwrap()
            .0
    }
}

// ============================================================
// HELPER: Normalize pixel data for training
// ============================================================

/// Convert a 28×28 u8 image to normalized f64 values in [0, 1].
pub fn normalize_pixels(pixels: &[u8]) -> Vec<f64> {
    pixels.iter().map(|&p| p as f64 / 255.0).collect()
}

/// Quantize pixel values into [0, INPUT_SCALE] for encrypted inference.
/// Maps [0, 255] → [0, INPUT_SCALE] linearly.
pub fn quantize_pixels(pixels: &[u8]) -> Vec<u64> {
    pixels
        .iter()
        .map(|&p| (p as u64 * INPUT_SCALE) / 255)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_network_forward() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let net = NetworkF64::new(&mut rng);
        let input = vec![0.5; INPUT_SIZE];
        let (hidden, logits) = net.forward(&input);
        assert_eq!(hidden.len(), HIDDEN_SIZE);
        assert_eq!(logits.len(), OUTPUT_SIZE);
    }

    #[test]
    fn test_quantize_preserves_structure() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let net = NetworkF64::new(&mut rng);
        let qnet = net.quantize();

        assert_eq!(qnet.w1.len(), HIDDEN_SIZE);
        assert_eq!(qnet.w1[0].len(), INPUT_SIZE);
        assert_eq!(qnet.w2.len(), OUTPUT_SIZE);
        assert_eq!(qnet.w2[0].len(), HIDDEN_SIZE);

        // All weights should be within bounds
        for row in &qnet.w1 {
            for &w in row {
                assert!(w.abs() <= WEIGHT_BOUND as i32);
            }
        }
    }

    #[test]
    fn test_quantized_inference_runs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let net = NetworkF64::new(&mut rng);
        let qnet = net.quantize();
        let input = vec![1u64; INPUT_SIZE];
        let output = qnet.infer_plaintext(&input);
        assert_eq!(output.len(), OUTPUT_SIZE);
    }
}
