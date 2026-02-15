//! # Data Generation Module
//!
//! Generates a training dataset for digit recognition by rendering
//! digits 0-9 using Windows system fonts into 28×28 grayscale images.
//!
//! Uses the `rusttype` crate to load TrueType fonts and render
//! individual glyph bitmaps. Each digit is centered in the 28×28 canvas.

use image::{GrayImage, Luma};
use rand::Rng;
use rusttype::{Font, Scale};
use std::io::Read;
use std::path::Path;

/// Image dimensions matching MNIST format.
pub const IMG_SIZE: usize = 28;

/// A single labeled sample: 28×28 grayscale pixels + digit label.
pub struct Sample {
    pub pixels: Vec<u8>, // Length: IMG_SIZE * IMG_SIZE = 784
    pub label: u8,       // 0-9
}

/// Scan Windows system fonts directory and return paths to .ttf files.
pub fn find_system_fonts() -> Vec<String> {
    let fonts_dir = Path::new("C:\\Windows\\Fonts");
    let mut font_paths = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fonts_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext
                    .to_str()
                    .map(|s| s.eq_ignore_ascii_case("ttf"))
                    .unwrap_or(false)
                {
                    if let Some(path_str) = path.to_str() {
                        font_paths.push(path_str.to_string());
                    }
                }
            }
        }
    }

    font_paths
}

/// Render a single digit using the given font at a specific scale into a 28×28 grayscale image.
///
/// The glyph is centered in the canvas. Returns None if the font doesn't have the glyph.
pub fn render_digit_at_scale(font: &Font, digit: u8, font_scale: f32) -> Option<Vec<u8>> {
    let ch = (b'0' + digit) as char;

    let scale = Scale::uniform(font_scale);
    let v_metrics = font.v_metrics(scale);
    let glyph = font.glyph(ch).scaled(scale);

    // Check that glyph exists
    let h_metrics = glyph.h_metrics();
    if h_metrics.advance_width < 1.0 {
        return None;
    }

    let glyph = glyph.positioned(rusttype::point(0.0, v_metrics.ascent));

    // Get bounding box
    let bb = glyph.pixel_bounding_box()?;
    let glyph_w = (bb.max.x - bb.min.x) as u32;
    let glyph_h = (bb.max.y - bb.min.y) as u32;

    if glyph_w == 0 || glyph_h == 0 {
        return None;
    }

    // Render glyph to a temporary buffer
    let mut glyph_buf = vec![0u8; (glyph_w * glyph_h) as usize];
    glyph.draw(|x, y, v| {
        let idx = (y * glyph_w + x) as usize;
        if idx < glyph_buf.len() {
            glyph_buf[idx] = (v * 255.0) as u8;
        }
    });

    // Create 28×28 canvas and center the glyph
    let mut img = GrayImage::new(IMG_SIZE as u32, IMG_SIZE as u32);
    let offset_x = ((IMG_SIZE as u32).saturating_sub(glyph_w)) / 2;
    let offset_y = ((IMG_SIZE as u32).saturating_sub(glyph_h)) / 2;

    for y in 0..glyph_h {
        for x in 0..glyph_w {
            let dst_x = offset_x + x;
            let dst_y = offset_y + y;
            if dst_x < IMG_SIZE as u32 && dst_y < IMG_SIZE as u32 {
                let val = glyph_buf[(y * glyph_w + x) as usize];
                img.put_pixel(dst_x, dst_y, Luma([val]));
            }
        }
    }

    Some(img.into_raw())
}

/// Render a digit at the default scale (20px).
pub fn render_digit(font: &Font, digit: u8) -> Option<Vec<u8>> {
    render_digit_at_scale(font, digit, 20.0)
}

/// Apply random augmentation with multiple transforms:
/// - Translation (±3px)
/// - Small rotation (±10°)
/// - Scale jitter (0.85x - 1.15x)
/// - Additive noise
/// - Random stroke thickness via erosion/dilation
pub fn augment(pixels: &[u8], rng: &mut impl Rng) -> Vec<u8> {
    let size = IMG_SIZE as i32;

    // Random parameters
    let shift_x: f64 = rng.gen_range(-3.0..=3.0);
    let shift_y: f64 = rng.gen_range(-3.0..=3.0);
    let angle: f64 = rng.gen_range(-12.0_f64..=12.0).to_radians();
    let scale: f64 = rng.gen_range(0.82..=1.18);
    let noise_level: u8 = rng.gen_range(0..=15);

    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let cx = IMG_SIZE as f64 / 2.0;
    let cy = IMG_SIZE as f64 / 2.0;

    let mut result = vec![0u8; IMG_SIZE * IMG_SIZE];

    for y in 0..size {
        for x in 0..size {
            // Inverse transform: from destination to source
            let dx = x as f64 - cx;
            let dy = y as f64 - cy;

            // Inverse scale
            let sx = dx / scale;
            let sy = dy / scale;

            // Inverse rotation
            let rx = sx * cos_a + sy * sin_a;
            let ry = -sx * sin_a + sy * cos_a;

            // Inverse translation + re-center
            let src_x = rx + cx - shift_x;
            let src_y = ry + cy - shift_y;

            // Bilinear interpolation
            let px = src_x.floor() as i32;
            let py = src_y.floor() as i32;
            let fx = src_x - px as f64;
            let fy = src_y - py as f64;

            let get = |px: i32, py: i32| -> f64 {
                if px >= 0 && px < size && py >= 0 && py < size {
                    pixels[py as usize * IMG_SIZE + px as usize] as f64
                } else {
                    0.0
                }
            };

            let val = get(px, py) * (1.0 - fx) * (1.0 - fy)
                + get(px + 1, py) * fx * (1.0 - fy)
                + get(px, py + 1) * (1.0 - fx) * fy
                + get(px + 1, py + 1) * fx * fy;

            // Add noise
            let noise = rng.gen_range(0..=noise_level) as i32 - (noise_level / 2) as i32;
            let final_val = (val as i32 + noise).clamp(0, 255) as u8;

            result[y as usize * IMG_SIZE + x as usize] = final_val;
        }
    }

    // Random morphological operation: thicken or thin strokes
    let morph = rng.gen_range(0..3);
    if morph == 1 {
        // Dilate (thicken)
        dilate(&result)
    } else if morph == 2 {
        // Erode (thin) — but only if there's enough content
        let sum: u32 = result.iter().map(|&p| p as u32).sum();
        if sum > 20000 {
            erode(&result)
        } else {
            result
        }
    } else {
        result
    }
}

/// Morphological dilation — thickens white strokes.
fn dilate(pixels: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; IMG_SIZE * IMG_SIZE];
    let size = IMG_SIZE as i32;
    for y in 0..size {
        for x in 0..size {
            let mut max_val = 0u8;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < size && ny >= 0 && ny < size {
                        max_val = max_val.max(pixels[ny as usize * IMG_SIZE + nx as usize]);
                    }
                }
            }
            out[y as usize * IMG_SIZE + x as usize] = max_val;
        }
    }
    out
}

/// Morphological erosion — thins white strokes.
fn erode(pixels: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; IMG_SIZE * IMG_SIZE];
    let size = IMG_SIZE as i32;
    for y in 0..size {
        for x in 0..size {
            let mut min_val = 255u8;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < size && ny >= 0 && ny < size {
                        min_val = min_val.min(pixels[ny as usize * IMG_SIZE + nx as usize]);
                    }
                }
            }
            out[y as usize * IMG_SIZE + x as usize] = min_val;
        }
    }
    out
}

/// List of popular Google Fonts families known to contain digit glyphs.
/// Downloaded from the google/fonts GitHub repository.
const GOOGLE_FONT_URLS: &[(&str, &str)] = &[
    ("Roboto-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf"),
    ("OpenSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/opensans/OpenSans%5Bwdth%2Cwght%5D.ttf"),
    ("Lato-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/lato/Lato-Regular.ttf"),
    ("Montserrat-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf"),
    ("Poppins-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf"),
    ("Nunito-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/nunito/Nunito%5Bwght%5D.ttf"),
    ("Ubuntu-Regular", "https://raw.githubusercontent.com/google/fonts/main/ufl/ubuntu/Ubuntu-Regular.ttf"),
    ("PTSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/ptsans/PT_Sans-Web-Regular.ttf"),
    ("SourceCodePro-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf"),
    ("NotoSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf"),
    ("Raleway-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/Raleway%5Bwght%5D.ttf"),
    ("Quicksand-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/quicksand/Quicksand%5Bwght%5D.ttf"),
    ("FiraSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/firasans/FiraSans-Regular.ttf"),
    ("WorkSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/worksans/WorkSans%5Bwght%5D.ttf"),
    ("Inconsolata-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/inconsolata/Inconsolata%5Bwdth%2Cwght%5D.ttf"),
    ("JetBrainsMono-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf"),
    ("Karla-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/karla/Karla%5Bwght%5D.ttf"),
    ("Bitter-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/bitter/Bitter%5Bwght%5D.ttf"),
    ("Cabin-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/cabin/Cabin%5Bwdth%2Cwght%5D.ttf"),
    ("Exo2-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/exo2/Exo2%5Bwght%5D.ttf"),
    ("DancingScript-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf"),
    ("Comfortaa-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf"),
    ("Barlow-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/barlow/Barlow-Regular.ttf"),
    ("Archivo-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/archivo/Archivo%5Bwdth%2Cwght%5D.ttf"),
    ("SpaceMono-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Regular.ttf"),
    ("OverpassMono-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/overpassmono/OverpassMono%5Bwght%5D.ttf"),
    ("IBMPlexSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexsans/IBMPlexSans-Regular.ttf"),
    ("Rubik-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/rubik/Rubik%5Bwght%5D.ttf"),
    ("Lexend-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/lexend/Lexend%5Bwght%5D.ttf"),
    ("Outfit-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/outfit/Outfit%5Bwght%5D.ttf"),
    ("Heebo-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/heebo/Heebo%5Bwght%5D.ttf"),
    ("Kanit-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/kanit/Kanit-Regular.ttf"),
    ("Manrope-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/manrope/Manrope%5Bwght%5D.ttf"),
    ("Catamaran-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/catamaran/Catamaran%5Bwght%5D.ttf"),
    ("Righteous-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/righteous/Righteous-Regular.ttf"),
    ("Rajdhani-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani%5Bwght%5D.ttf"),
    ("Jost-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/jost/Jost%5Bwght%5D.ttf"),
    ("Signika-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/signika/Signika%5Bwght%5D.ttf"),
    ("FredokaOne-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/fredoka/Fredoka%5Bwdth%2Cwght%5D.ttf"),
    ("Overpass-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/overpass/Overpass%5Bwght%5D.ttf"),
    ("AbrilFatface-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/abrilfatface/AbrilFatface-Regular.ttf"),
    ("BarlowCondensed-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-Regular.ttf"),
    ("SpaceGrotesk-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf"),
    ("AlegreyaSans-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/alegreyasans/AlegreyaSans-Regular.ttf"),
    ("Arimo-Regular", "https://raw.githubusercontent.com/google/fonts/main/ofl/arimo/Arimo%5Bwght%5D.ttf"),
    ("Cousine-Regular", "https://raw.githubusercontent.com/google/fonts/main/apache/cousine/Cousine-Regular.ttf"),
    ("Tinos-Regular", "https://raw.githubusercontent.com/google/fonts/main/apache/tinos/Tinos-Regular.ttf"),
];

/// Download Google Fonts TTF files to a local cache directory.
/// Returns paths to downloaded font files. Skips already-cached fonts.
pub fn download_google_fonts() -> Vec<String> {
    let cache_dir = Path::new("fonts_cache");
    if !cache_dir.exists() {
        let _ = std::fs::create_dir(cache_dir);
    }

    let mut paths = Vec::new();
    let total = GOOGLE_FONT_URLS.len();

    for (i, (name, url)) in GOOGLE_FONT_URLS.iter().enumerate() {
        let file_path = cache_dir.join(format!("{}.ttf", name));

        if file_path.exists() {
            // Already cached
            if let Some(p) = file_path.to_str() {
                paths.push(p.to_string());
            }
            continue;
        }

        print!("  Downloading [{}/{}] {}...", i + 1, total, name);

        match ureq::get(url).call() {
            Ok(resp) => {
                let mut bytes = Vec::new();
                if resp.into_reader().read_to_end(&mut bytes).is_ok() && !bytes.is_empty() {
                    if std::fs::write(&file_path, &bytes).is_ok() {
                        println!(" ✓ ({}KB)", bytes.len() / 1024);
                        if let Some(p) = file_path.to_str() {
                            paths.push(p.to_string());
                        }
                    } else {
                        println!(" ✗ (write error)");
                    }
                } else {
                    println!(" ✗ (empty response)");
                }
            }
            Err(e) => {
                println!(" ✗ ({})", e);
            }
        }
    }

    println!("  Google Fonts: {} cached", paths.len());
    paths
}

/// Generate the full training dataset from system fonts + Google Fonts.
///
/// For each font, renders digits 0-9
/// at multiple scales and creates augmented copies.
///
/// Typical output: ~5000-50000 samples depending on available fonts.
pub fn generate_dataset(augmentations_per_sample: usize) -> Vec<Sample> {
    // Collect fonts from both sources
    let mut font_paths = find_system_fonts();
    println!("Found {} .ttf system fonts", font_paths.len());

    // Download Google Fonts (cached)
    println!("Loading Google Fonts...");
    let google_paths = download_google_fonts();
    font_paths.extend(google_paths);

    let mut dataset = Vec::new();
    let mut rng = rand::thread_rng();
    let mut fonts_used = 0usize;

    // Render at multiple scales to get different stroke widths
    let scales = [16.0, 18.0, 20.0, 22.0, 24.0];

    for path in &font_paths {
        let font_data = match std::fs::read(path) {
            Ok(data) => data,
            Err(_) => continue,
        };

        let font = match Font::try_from_vec(font_data) {
            Some(f) => f,
            None => continue,
        };

        let mut font_has_digits = false;

        for digit in 0..10u8 {
            for &scale in &scales {
                if let Some(pixels) = render_digit_at_scale(&font, digit, scale) {
                    font_has_digits = true;

                    // Add original
                    dataset.push(Sample {
                        pixels: pixels.clone(),
                        label: digit,
                    });

                    // Add augmented copies
                    for _ in 0..augmentations_per_sample {
                        let aug = augment(&pixels, &mut rng);
                        dataset.push(Sample {
                            pixels: aug,
                            label: digit,
                        });
                    }
                }
            }
        }

        if font_has_digits {
            fonts_used += 1;
        }
    }

    println!(
        "Generated {} samples from {} fonts ({} scales × {} augmentations each)",
        dataset.len(),
        fonts_used,
        scales.len(),
        augmentations_per_sample
    );

    dataset
}

/// Center-normalize a grayscale image to 28x28 (MNIST-style).
///
/// 1. Find bounding box of non-black pixels (threshold > 20)
/// 2. Crop to bounding box
/// 3. Scale to fit in ~20x20 area (preserving aspect ratio)
/// 4. Center on 28x28 canvas
fn center_normalize_image(img: &image::GrayImage) -> Vec<u8> {
    let (w, h) = (img.width(), img.height());
    let threshold = 20u8;

    // Find bounding box
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut has_content = false;

    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y)[0] > threshold {
                has_content = true;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }

    if !has_content {
        return vec![0u8; IMG_SIZE * IMG_SIZE];
    }

    // Crop
    let crop_w = max_x - min_x + 1;
    let crop_h = max_y - min_y + 1;
    let cropped = image::imageops::crop_imm(img, min_x, min_y, crop_w, crop_h).to_image();

    // Scale to fit within 20x20 (preserving aspect ratio)
    let target_size = 20u32;
    let scale = target_size as f64 / crop_w.max(crop_h) as f64;
    let scaled_w = (crop_w as f64 * scale).round().max(1.0) as u32;
    let scaled_h = (crop_h as f64 * scale).round().max(1.0) as u32;
    let scaled = image::imageops::resize(
        &cropped,
        scaled_w,
        scaled_h,
        image::imageops::FilterType::Lanczos3,
    );

    // Center on 28x28
    let offset_x = (IMG_SIZE as u32 - scaled_w) / 2;
    let offset_y = (IMG_SIZE as u32 - scaled_h) / 2;

    let mut result = image::GrayImage::new(IMG_SIZE as u32, IMG_SIZE as u32);
    image::imageops::overlay(&mut result, &scaled, offset_x as i64, offset_y as i64);

    result.into_raw()
}

/// Load images from a directory structure where filenames are "{label}_{idx}.png".
/// Returns a vector of Samples.
pub fn load_images_from_dir(path: &Path) -> Vec<Sample> {
    let mut samples = Vec::new();
    if !path.exists() {
        println!("Warning: Directory {} does not exist.", path.display());
        return samples;
    }

    println!("Scanning {}...", path.display());
    let mut count = 0;

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Expected filename: "5_123.png"
            if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some(label_str) = file_stem.split('_').next() {
                    if let Ok(label) = label_str.parse::<u8>() {
                        if let Ok(img) = image::open(&path) {
                            let gray = img.into_luma8();
                            // MNIST images are already 28x28 and centered, but we can ensure it
                            // or just use raw pixels if we trust the extraction.
                            // The extraction script saved them as 28x28 grayscale.

                            // Check if dimensions match IMG_SIZE
                            if gray.width() == IMG_SIZE as u32 && gray.height() == IMG_SIZE as u32 {
                                samples.push(Sample {
                                    pixels: gray.into_raw(),
                                    label,
                                });
                            } else {
                                // Fallback: resize/center if needed (shouldn't be for this dataset)
                                let centered = center_normalize_image(&gray);
                                samples.push(Sample {
                                    pixels: centered,
                                    label,
                                });
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    println!("  Loaded {} samples from {}", count, path.display());
    samples
}

/// Load the full MNIST dataset from `static/mnist_images/train` and `test`.
pub fn load_mnist_flat() -> Vec<Sample> {
    let base = Path::new("static/mnist_images");
    let mut samples = load_images_from_dir(&base.join("train"));
    samples.extend(load_images_from_dir(&base.join("test")));
    samples
}

/// Load custom dataset from "static/my_assets" directory.
/// Files should be named like "label_timestamp.png".
/// Returns a vector of Samples with center-normalized images.
pub fn load_custom_dataset() -> Vec<Sample> {
    // Reuse the generic loader but with augmentation
    let dataset_dir = Path::new("static/my_assets");
    let raw_samples = load_images_from_dir(dataset_dir);

    // Augment custom samples heavily since they are few but valuable
    // (create 20 copies of each user drawing)
    let mut augmented_samples = Vec::new();
    let mut rng = rand::thread_rng();

    for sample in &raw_samples {
        // Add original
        augmented_samples.push(Sample {
            pixels: sample.pixels.clone(),
            label: sample.label,
        });

        // Add augmented copies
        for _ in 0..20 {
            let aug = augment(&sample.pixels, &mut rng);
            augmented_samples.push(Sample {
                pixels: aug,
                label: sample.label,
            });
        }
    }

    println!(
        "Loaded {} custom samples (expanded to {})",
        raw_samples.len(),
        augmented_samples.len()
    );
    augmented_samples
}

/// Convert dataset to the format expected by the neural network trainer.
/// Returns Vec<(f64 pixels, label)> with pixels normalized to [0, 1].
pub fn dataset_to_training(samples: &[Sample]) -> Vec<(Vec<f64>, u8)> {
    samples
        .iter()
        .map(|s| {
            let pixels: Vec<f64> = s.pixels.iter().map(|&p| p as f64 / 255.0).collect();
            (pixels, s.label)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_fonts() {
        let fonts = find_system_fonts();
        // Windows should have at least a few TTF fonts
        println!("Found {} TTF fonts", fonts.len());
        // Don't assert > 0 in case of CI environment, just log
    }

    #[test]
    fn test_augment_size_preserved() {
        let mut rng = rand::thread_rng();
        let pixels = vec![128u8; IMG_SIZE * IMG_SIZE];
        let result = augment(&pixels, &mut rng);
        assert_eq!(result.len(), IMG_SIZE * IMG_SIZE);
    }
}
