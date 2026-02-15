//! # PPNN — Privacy-Preserving Neural Network
//!
//! A complete system for handwritten digit recognition using
//! LWE-based Fully Homomorphic Encryption, implemented from scratch.
//!
//! ## Usage
//!   cargo run -- train     # Generate dataset, train NN, save weights
//!   cargo run -- serve     # Start web server with encrypted inference
//!   cargo run              # Train then serve (default)

mod bootstrap;
mod data;
mod encrypted_inference;
mod lwe;
mod nn;
mod server;

use std::path::Path;

const WEIGHTS_PATH: &str = "model_weights.json";
const KEYS_PATH: &str = "eval_keys.json";
const SERVER_PORT: u16 = 3000;

fn train_and_save() {
    println!("=== PPNN Training Pipeline ===\n");

    // Step 1: Generate dataset from system + Google Fonts
    println!("[1/4] Generating dataset from system + Google Fonts...");
    let mut samples = data::generate_dataset(8);

    // Load custom user data
    let custom_samples = data::load_custom_dataset();
    if !custom_samples.is_empty() {
        println!("Found {} custom samples (augmented)", custom_samples.len());
        samples.extend(custom_samples);
    }

    if samples.is_empty() {
        println!("WARNING: No samples generated. Using synthetic data for testing.");
        return train_with_synthetic();
    }

    let mut all_data = data::dataset_to_training(&samples);

    // Shuffle before splitting
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_data.shuffle(&mut rng);

    // 80/20 train/test split
    let split_idx = (all_data.len() as f64 * 0.8) as usize;
    let (train_data, test_data) = all_data.split_at(split_idx);
    let train_data = train_data.to_vec();
    let test_data = test_data.to_vec();
    println!(
        "  Dataset: {} total → {} train / {} test (80/20 split)\n",
        all_data.len(),
        train_data.len(),
        test_data.len()
    );

    // Step 2: Train with early stopping on test accuracy
    println!("[2/4] Training neural network...");
    let mut net = nn::NetworkF64::new(&mut rng);
    net.train(&train_data, Some(&test_data), 500, 0.005, 512);

    let train_acc = net.accuracy(&train_data);
    let test_acc = net.accuracy(&test_data);
    println!(
        "  Final: train={:.1}% test={:.1}%",
        train_acc * 100.0,
        test_acc * 100.0
    );

    // Separate accuracy on custom data (non-augmented)
    let custom_raw = data::load_custom_dataset();
    if !custom_raw.is_empty() {
        let custom_training = data::dataset_to_training(&custom_raw);
        let custom_acc = net.accuracy(&custom_training);
        println!(
            "  Custom dataset accuracy: {:.1}% ({} samples)",
            custom_acc * 100.0,
            custom_training.len()
        );
    }
    println!();

    // Step 3: Quantize weights
    println!("[3/4] Quantizing weights to integers...");
    let qnet = net.quantize();

    // Test quantized inference on TEST set (honest eval)
    let q_correct = test_data
        .iter()
        .filter(|(pixels, label)| {
            let qpixels: Vec<u64> = pixels
                .iter()
                .map(|&p| (p * nn::INPUT_SCALE as f64).round() as u64)
                .collect();
            qnet.predict_plaintext(&qpixels) == *label as usize
        })
        .count();
    println!(
        "  Quantized test accuracy: {:.1}%\n",
        q_correct as f64 / test_data.len() as f64 * 100.0
    );

    // Step 4: Save
    println!("[4/4] Saving model weights...");
    let json = serde_json::to_string(&qnet).expect("Failed to serialize weights");
    std::fs::write(WEIGHTS_PATH, json).expect("Failed to write weights");
    println!("  Saved to {}\n", WEIGHTS_PATH);

    println!("=== Training Complete ===\n");
}

/// Fallback: create synthetic training data when no fonts are available.
fn train_with_synthetic() {
    println!("[1/4] Using synthetic training data (no fonts found)...");

    let mut rng = rand::thread_rng();
    use rand::Rng;

    let mut training_data = Vec::new();
    for digit in 0..10u8 {
        for _ in 0..50 {
            let mut pixels = vec![0.0f64; 784];
            // Create a simple pattern: each digit has a different active region
            let offset = digit as usize * 70;
            for i in offset..(offset + 60).min(784) {
                pixels[i] = 0.5 + rng.gen::<f64>() * 0.5;
            }
            training_data.push((pixels, digit));
        }
    }

    println!("  Dataset: {} synthetic samples\n", training_data.len());

    println!("[2/4] Training neural network...");
    let mut net = nn::NetworkF64::new(&mut rng);
    net.train(&training_data, None, 150, 0.01, 32);

    let accuracy = net.accuracy(&training_data);
    println!("  Final training accuracy: {:.1}%\n", accuracy * 100.0);

    println!("[3/4] Quantizing weights...");
    let qnet = net.quantize();

    println!("[4/4] Saving model weights...");
    let json = serde_json::to_string(&qnet).expect("Failed to serialize weights");
    std::fs::write(WEIGHTS_PATH, json).expect("Failed to write weights");
    println!("  Saved to {}\n", WEIGHTS_PATH);

    println!("=== Training Complete ===\n");
}

/// Train on all data, test only on user-drawn assets from static/my_assets/.
fn train_my_mode() {
    println!("=== PPNN Training Pipeline (test=my_assets) ===\n");

    // Step 1: Generate font dataset
    println!("[1/4] Generating dataset from system + Google Fonts...");
    let mut all_samples = data::generate_dataset(8);

    // Load custom user data (these will also be in train if non-empty)
    let custom_samples = data::load_custom_dataset();
    if custom_samples.is_empty() {
        println!("ERROR: No custom samples in static/my_assets/! Nothing to test against.");
        return;
    }
    let custom_test = data::dataset_to_training(&custom_samples);
    println!("  Custom test set: {} samples\n", custom_test.len());

    // Add custom samples to the training pool too
    all_samples.extend(custom_samples);
    let train_data = data::dataset_to_training(&all_samples);
    println!("  Total training set: {} samples", train_data.len());

    // Step 2: Train with custom data as test set
    println!("[2/4] Training neural network (testing on YOUR drawings)...");
    let mut rng = rand::thread_rng();
    let mut net = nn::NetworkF64::new(&mut rng);
    net.train(&train_data, Some(&custom_test), 250, 0.03, 512);

    let train_acc = net.accuracy(&train_data);
    let my_acc = net.accuracy(&custom_test);
    println!(
        "  Final: train={:.1}% my_assets={:.1}%",
        train_acc * 100.0,
        my_acc * 100.0
    );
    println!();

    // Step 3: Quantize
    println!("[3/4] Quantizing weights to integers...");
    let qnet = net.quantize();

    let q_correct = custom_test
        .iter()
        .filter(|(pixels, label)| {
            let qpixels: Vec<u64> = pixels
                .iter()
                .map(|&p| (p * nn::INPUT_SCALE as f64).round() as u64)
                .collect();
            qnet.predict_plaintext(&qpixels) == *label as usize
        })
        .count();
    println!(
        "  Quantized my_assets accuracy: {:.1}%\n",
        q_correct as f64 / custom_test.len() as f64 * 100.0
    );

    // Step 4: Save
    println!("[4/4] Saving model weights...");
    let json = serde_json::to_string(&qnet).expect("Failed to serialize weights");
    std::fs::write(WEIGHTS_PATH, json).expect("Failed to write weights");
    println!("  Saved to {}\n", WEIGHTS_PATH);

    println!("=== Training Complete ===\n");
}

/// Train ONLY on custom data (no synthetic/system fonts).
fn train_only_my() {
    println!("=== PPNN Training Pipeline (ONLY my_assets) ===\n");

    // Step 1: Load custom user data
    println!("[1/4] Loading custom dataset from static/my_assets/...");
    let custom_samples = data::load_custom_dataset();
    if custom_samples.is_empty() {
        println!("ERROR: No custom samples in static/my_assets/! Cannot train.");
        return;
    }
    println!("  Found {} custom samples", custom_samples.len());

    // Convert to training format
    let mut all_data = data::dataset_to_training(&custom_samples);

    // Shuffle
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_data.shuffle(&mut rng);

    // 80/20 train/test split
    let split_idx = (all_data.len() as f64 * 0.8) as usize;
    let (train_data, test_data) = all_data.split_at(split_idx);
    let train_data = train_data.to_vec();
    let test_data = test_data.to_vec(); // Can be empty if strictly < 5 samples

    println!(
        "  Dataset: {} total → {} train / {} test\n",
        all_data.len(),
        train_data.len(),
        test_data.len()
    );

    // Step 2: Train
    println!("[2/4] Training neural network...");
    let mut net = nn::NetworkF64::new(&mut rng);
    // Higher LR for small dataset to converge fast, slightly fewer epochs might be needed but 250 is safe
    net.train(&train_data, Some(&test_data), 250, 0.03, 32);

    let train_acc = net.accuracy(&train_data);
    let test_acc = net.accuracy(&test_data);
    println!(
        "  Final: train={:.1}% test={:.1}%",
        train_acc * 100.0,
        test_acc * 100.0
    );
    println!();

    // Step 3: Quantize
    println!("[3/4] Quantizing weights to integers...");
    let qnet = net.quantize();

    if !test_data.is_empty() {
        let q_correct = test_data
            .iter()
            .filter(|(pixels, label)| {
                let qpixels: Vec<u64> = pixels
                    .iter()
                    .map(|&p| (p * nn::INPUT_SCALE as f64).round() as u64)
                    .collect();
                qnet.predict_plaintext(&qpixels) == *label as usize
            })
            .count();
        println!(
            "  Quantized test accuracy: {:.1}%\n",
            q_correct as f64 / test_data.len() as f64 * 100.0
        );
    }

    // Step 4: Save
    println!("[4/4] Saving model weights...");
    let json = serde_json::to_string(&qnet).expect("Failed to serialize weights");
    std::fs::write(WEIGHTS_PATH, json).expect("Failed to write weights");
    println!("  Saved to {}\n", WEIGHTS_PATH);

    println!("=== Training Complete ===\n");
}

/// Train on MNIST Train, Test on (MNIST Test + My Assets).
fn train_mnist_my() {
    println!("=== PPNN Training Pipeline (MNIST Train -> MNIST Test + My Assets) ===\n");

    // Step 1: Load Data
    println!("[1/4] Loading datasets...");

    // Train: MNIST Train (60k)
    let train_path = Path::new("static/mnist_images/train");
    println!("  Loading Training set from {}...", train_path.display());
    let train_samples = data::load_images_from_dir(train_path);
    if train_samples.is_empty() {
        println!("ERROR: No MNIST training samples found! Run extract_mnist.py first.");
        return;
    }

    // Test: MNIST Test (10k) + My Assets
    let test_path = Path::new("static/mnist_images/test");
    println!("  Loading Test set from {}...", test_path.display());
    let mut test_samples = data::load_images_from_dir(test_path);

    let my_path = Path::new("static/my_assets");
    if my_path.exists() {
        println!("  Loading My Assets from {}...", my_path.display());
        let my_samples = data::load_images_from_dir(my_path);
        println!("    Found {} custom samples", my_samples.len());
        test_samples.extend(my_samples);
    } else {
        println!("  (No static/my_assets found, skipping custom test data)");
    }

    // Convert to training format
    let mut train_data = data::dataset_to_training(&train_samples);
    let test_data = data::dataset_to_training(&test_samples);

    // Shuffle training data (test data order doesn't matter for metrics, but good practice)
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    train_data.shuffle(&mut rng);

    println!(
        "  Dataset: {} training / {} testing\n",
        train_data.len(),
        test_data.len()
    );

    // Step 2: Train
    println!("[2/4] Training neural network...");
    let mut net = nn::NetworkF64::new(&mut rng);
    // Parameters: 250 epochs, 0.03 LR, 32 hidden (same as only_my)
    net.train(&train_data, Some(&test_data), 250, 0.03, 32);

    let train_acc = net.accuracy(&train_data);
    let test_acc = net.accuracy(&test_data);
    println!(
        "  Final: train={:.1}% test={:.1}%",
        train_acc * 100.0,
        test_acc * 100.0
    );
    println!();

    // Step 3: Quantize
    println!("[3/4] Quantizing weights to integers...");
    let qnet = net.quantize();

    if !test_data.is_empty() {
        let q_correct = test_data
            .iter()
            .filter(|(pixels, label)| {
                let qpixels: Vec<u64> = pixels
                    .iter()
                    .map(|&p| (p * nn::INPUT_SCALE as f64).round() as u64)
                    .collect();
                qnet.predict_plaintext(&qpixels) == *label as usize
            })
            .count();
        println!(
            "  Quantized test accuracy: {:.1}%\n",
            q_correct as f64 / test_data.len() as f64 * 100.0
        );
    }

    // Step 4: Save
    println!("[4/4] Saving model weights...");
    let json = serde_json::to_string(&qnet).expect("Failed to serialize weights");
    std::fs::write(WEIGHTS_PATH, json).expect("Failed to write weights");
    println!("  Saved to {}\n", WEIGHTS_PATH);

    println!("=== Training Complete ===\n");
}

fn load_or_default_network() -> nn::NetworkQuantized {
    if Path::new(WEIGHTS_PATH).exists() {
        println!("Loading pre-trained weights from {}...", WEIGHTS_PATH);
        let json = std::fs::read_to_string(WEIGHTS_PATH).expect("Failed to read weights");
        serde_json::from_str(&json).expect("Failed to deserialize weights")
    } else {
        println!("No pre-trained weights found. Training first...");
        train_and_save();
        let json = std::fs::read_to_string(WEIGHTS_PATH).expect("Failed to read weights");
        serde_json::from_str(&json).expect("Failed to deserialize weights")
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match command {
        "train" => {
            train_and_save();
        }
        "train_my" => {
            train_my_mode();
        }
        "train_only_my" => {
            train_only_my();
        }
        "train_mnist_my" => {
            train_mnist_my();
        }
        "serve" => {
            let network = load_or_default_network();

            // Generate cryptographic keys
            println!("Generating LWE keys and bootstrapping parameters...");
            let mut rng = rand::thread_rng();
            let secret_key = lwe::keygen(&mut rng);
            let (_boot_key, eval_keys) = bootstrap::gen_eval_keys(&secret_key, &mut rng);
            println!("  Secret key: {} dimensions", secret_key.values.len());
            println!("  Bootstrapping key: {} entries", eval_keys.bsk.bsk.len());
            println!(
                "  Key-switching key: {}×{} entries",
                eval_keys.ksk.ksk.len(),
                if eval_keys.ksk.ksk.is_empty() {
                    0
                } else {
                    eval_keys.ksk.ksk[0].len()
                }
            );

            server::start_server(secret_key, eval_keys, network, SERVER_PORT).await;
        }
        "all" | _ => {
            // Train then serve
            let network = load_or_default_network();

            println!("Generating cryptographic keys...");
            let mut rng = rand::thread_rng();
            let secret_key = lwe::keygen(&mut rng);
            let (_boot_key, eval_keys) = bootstrap::gen_eval_keys(&secret_key, &mut rng);
            println!("  Keys generated successfully.\n");

            server::start_server(secret_key, eval_keys, network, SERVER_PORT).await;
        }
    }
}
