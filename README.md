# PPNN (Privacy-Preserving Neural Network)

**A fully homomorphic neural network built from scratch in Rust.**

<div align="center">

<video src="[https://github.com/user-attachments/assets/a878c12a-7e6b-4b02-b662-cae0a6c319fa](https://github.com/user-attachments/assets/a878c12a-7e6b-4b02-b662-cae0a6c319fa)" controls="controls" style="max-width: 700px;">
</video>

<h3>QAT Training Analysis</h3>
<img src="assets/qat_analysis.png" alt="QAT Training Analysis" width="700">

<div align="left" style="max-width: 700px; margin: 0 auto;">
<p>
<b>Training Efficiency Comparison (Quantization Aware Training):</b>
</p>
<ul>
<li><b>Green</b>: Training on a mixed dataset (My Assets + Fonts). Slow convergence.</li>
<li><b>Red</b> (LR=0.03) & <b>Blue</b> (LR=0.06): Training on the same mixed dataset but with an increased Learning Rate. Reaches a plateau faster.</li>
<li><b>Yellow</b> (LR=0.03) & <b>Purple</b> (LR=0.06): <code>train_only_my</code> mode. Training <b>only</b> on the user dataset. Instant convergence and high accuracy for specific handwriting.</li>
</ul>
</div>

<img src="assets/architecture_diagram.png" alt="PPNN Architecture" width="700">

</div>

---

**PPNN** is a full implementation (from scratch, in Rust) of a handwritten digit recognition system operating on top of Fully Homomorphic Encryption (FHE).

The server **never** sees the digit you drew. It receives encrypted pixels, performs calculations (matrix multiplication + activation via bootstrapping), and returns an encrypted result. Only you (the client in the browser) can decrypt the response.

---

## Architecture and Security

The project implements the **TFHE (Torus FHE)** scheme with programmable bootstrapping.

### Implementation Details

In this demo version, the server technically **has access** to the private key (it is generated when the server starts); however, **all computations (inference)** on user data are performed strictly in encrypted form using homomorphic operations (addition, multiplication by a constant, bootstrapping).

The server does not "peek" into the data during calculations; instead, it performs mathematical operations on "noise," which turns into a meaningful answer only after decryption on the client side.

### Components

1. **LWE Crypto Engine**: Custom implementation in Rust (`src/lwe.rs`, `src/bootstrap.rs`).
2. **Neural Network**: A two-layer network trained on `f64` and quantized to `u64`.
3. **Collector**: A tool for creating your own dataset (`static/collector.html`).

---

## Launch and Usage

The project is fully ready to work "out of the box" — model weights are already included in the repository (`model_weights.json`).

### 1. Quick Start (Inference)

Run the server with the pre-trained model:

```bash
cargo run --release -- serve

```

Open `http://localhost:3000` in your browser. Draw digits, and the server will guess them in encrypted form.

### 2. Collecting Your Own Dataset

Want the network to recognize your specific handwriting?

1. Open the `static/collector.html` file in any browser (simply drag the file or use Live Server).
2. Draw digits in the square.
3. Press a number key on your keyboard (0-9) to save the sample.
4. Data is saved to the `static/my_assets/` folder.

### 3. Training on Your Own Data

After collecting the dataset, run the special training mode:

```bash
cargo run --release -- train_only_my

```

This will retrain the model **only** on your collected samples (very fast) and overwrite `model_weights.json`. After that, restart the server (`serve`) to apply the new model.

### 4. Full Training

If you need to retrain the network from scratch using a combination of system fonts and your data:

```bash
cargo run --release -- train

```

---

## Technical Details (Solutions & Fixes)

* **Rust**: Core logic, multithreading (`rayon`), HTTP server (`axum`).
* **TFHE/LWE**: Full custom implementation of crypto logic.
* **BigInt**: Client-side cryptography in pure JavaScript.
* **Neural Network**: Custom `Vec<f64>` / `Vec<u64>` implementation (no Torch/TensorFlow dependencies).
