//! # Web Server Module
//!
//! Axum-based HTTP server for the PPNN demo:
//!   - GET /         → Serve the HTML5 Canvas frontend
//!   - GET /key      → Return the LWE secret key (for demo enc/dec in JS)
//!   - GET /params   → Return LWE parameters (n, Q, t, Δ)
//!   - POST /infer   → Receive encrypted input, run encrypted inference, return result

use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::bootstrap::EvalKeys;
use crate::encrypted_inference::encrypted_infer;
use crate::lwe::*;
use crate::nn::NetworkQuantized;
use image::{GrayImage, Luma};
use std::time::{SystemTime, UNIX_EPOCH}; // For timestamps

// ============================================================
// API TYPES
// ============================================================

/// LWE parameters returned to the client.
/// Delta and t are sent as strings because u64 values exceed JS Number.MAX_SAFE_INTEGER.
#[derive(Serialize)]
pub struct LweParams {
    pub n: usize,
    pub q_bits: u32,
    pub t: String,
    pub delta: String,
}

/// Secret key response (for demo — client uses this to encrypt/decrypt).
/// Key values are 0 or 1, so they fit in JSON numbers.
#[derive(Serialize)]
pub struct KeyResponse {
    pub secret_key: Vec<u64>,
}

/// A ciphertext sent over the wire: (a[], b).
/// u64 values are serialized as decimal strings because JSON numbers
/// lose precision beyond 2^53 (JS Number.MAX_SAFE_INTEGER).
#[derive(Serialize, Deserialize, Clone)]
pub struct CiphertextWire {
    pub a: Vec<String>,
    pub b: String,
}

impl From<LweCiphertext> for CiphertextWire {
    fn from(ct: LweCiphertext) -> Self {
        CiphertextWire {
            a: ct.a.iter().map(|v| v.to_string()).collect(),
            b: ct.b.to_string(),
        }
    }
}

impl From<CiphertextWire> for LweCiphertext {
    fn from(w: CiphertextWire) -> Self {
        LweCiphertext {
            a: w.a.iter().map(|s| s.parse::<u64>().unwrap_or(0)).collect(),
            b: w.b.parse::<u64>().unwrap_or(0),
        }
    }
}

/// Inference request: 784 encrypted pixel ciphertexts.
#[derive(Deserialize)]
pub struct InferRequest {
    pub ciphertexts: Vec<CiphertextWire>,
}

/// Inference response: 10 encrypted output ciphertexts.
#[derive(Serialize)]
pub struct InferResponse {
    pub results: Vec<CiphertextWire>,
}

/// Collection request: label + 28x28 grayscale pixels
#[derive(Deserialize)]
pub struct CollectRequest {
    pub label: u8,
    pub pixels: Vec<u8>,
}

// ============================================================
// SERVER STATE
// ============================================================

/// Shared state holding the secret key, eval keys, and trained network.
pub struct AppState {
    pub secret_key: LweSecretKey,
    pub eval_keys: EvalKeys,
    pub network: NetworkQuantized,
}

// ============================================================
// HANDLERS
// ============================================================

/// GET / → Serve the HTML frontend.
async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// GET /params → Return LWE parameters.
async fn params_handler() -> Json<LweParams> {
    Json(LweParams {
        n: N_LWE,
        q_bits: 64,
        t: T_PLAINTEXT.to_string(),
        delta: DELTA.to_string(),
    })
}

/// GET /key → Return the secret key for client-side encrypt/decrypt.
/// WARNING: This is for demo purposes only. In production FHE,
/// the secret key NEVER leaves the client.
async fn key_handler(State(state): State<Arc<RwLock<AppState>>>) -> Json<KeyResponse> {
    let st = state.read().await;
    Json(KeyResponse {
        secret_key: st.secret_key.values.clone(),
    })
}

/// POST /infer → Run encrypted inference.
/// Receives 784 ciphertexts, returns 10 ciphertexts.
async fn infer_handler(
    State(state): State<Arc<RwLock<AppState>>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<InferResponse>, (StatusCode, String)> {
    if req.ciphertexts.len() != 784 {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Expected 784 ciphertexts, got {}", req.ciphertexts.len()),
        ));
    }

    let st = state.read().await;

    // Convert wire format to internal ciphertexts
    let input: Vec<LweCiphertext> = req.ciphertexts.into_iter().map(|w| w.into()).collect();

    println!(
        "[Server] Running encrypted inference on {} ciphertexts...",
        input.len()
    );

    // Run encrypted inference (this is the expensive part)
    let output = encrypted_infer(&input, &st.network, &st.eval_keys);

    println!(
        "[Server] Inference complete. Returning {} encrypted results.",
        output.len()
    );

    // Convert back to wire format
    let results: Vec<CiphertextWire> = output
        .into_iter()
        .map(|ct: LweCiphertext| ct.into())
        .collect();

    Ok(Json(InferResponse { results }))
}

/// POST /collect → Save a labeled sample to static/my_assets/
async fn collect_handler(Json(req): Json<CollectRequest>) -> StatusCode {
    if req.pixels.len() != 784 {
        return StatusCode::BAD_REQUEST;
    }

    // Ensure directory exists
    let dir = std::path::Path::new("static/my_assets");
    if !dir.exists() {
        let _ = std::fs::create_dir(dir);
    }

    // Generate filename: "label_timestamp_rand.png"
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let rand_suffix: u32 = rand::random();
    let filename = format!("{}_{}_{}.png", req.label, timestamp, rand_suffix);
    let path = dir.join(filename);

    // Create image and save
    let mut img = GrayImage::new(28, 28);
    for y in 0..28 {
        for x in 0..28 {
            let val = req.pixels[y * 28 + x];
            img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    match img.save(&path) {
        Ok(_) => {
            println!("[Server] Saved custom sample: {:?}", path);
            StatusCode::OK
        }
        Err(e) => {
            eprintln!("[Server] Failed to save sample: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

// ============================================================
// SERVER STARTUP
// ============================================================

/// Start the Axum web server.
pub async fn start_server(
    secret_key: LweSecretKey,
    eval_keys: EvalKeys,
    network: NetworkQuantized,
    port: u16,
) {
    let state = Arc::new(RwLock::new(AppState {
        secret_key,
        eval_keys,
        network,
    }));

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/params", get(params_handler))
        .route("/key", get(key_handler))
        .route("/infer", post(infer_handler))
        .route("/collect", post(collect_handler))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // 64MB for 784 u64 ciphertexts
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("\n========================================");
    println!("  PPNN Server running on http://127.0.0.1:{}", port);
    println!("  Open in browser to draw & classify digits");
    println!("========================================\n");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server error");
}
