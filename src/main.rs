mod burst;
mod align;
mod merge;
mod demosaic;
mod color_correction;
mod metadata;

use std::path::PathBuf;
use burst::BurstFrame;
use clap::Parser;

use crate::demosaic::demosaic;

#[derive(Parser)]
#[command(about = "HDR+ burst pipeline in Rust")]
struct Args {
    /// Directory containing the RAW burst files (sorted alphabetically = capture order)
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output file path for merged result
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    // Collect all RAW files in the directory, sorted by name
    let mut raw_paths: Vec<PathBuf> = std::fs::read_dir(&args.input_dir)
        .expect("Could not read input directory")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            // rawloader supports DNG, CR2, NEF, ARW, etc.
            let ext = path.extension()?.to_str()?.to_lowercase();
            if matches!(ext.as_str(), "dng" | "cr2" | "nef" | "arw" | "rw2") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    raw_paths.sort(); // For most burst naming schemes, alphabetical order = capture order

    if raw_paths.len() < 2 {
        eprintln!("Need at least 2 RAW files for a burst. Found {}", raw_paths.len());
        std::process::exit(1);
    }

    println!("Loading {} frames...", raw_paths.len());

    // Load all frames (frame 0 is the reference)
    let frames: Vec<BurstFrame> = raw_paths
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let path_str = path.to_str()
                .unwrap_or_else(|| panic!("Non-UTF8 path: {}", path.display()));
            let raw = rawloader::decode_file(path_str)
                .unwrap_or_else(|e| panic!("Failed to decode {}: {}", path.display(), e));
            BurstFrame::from_raw_image(&raw, path_str, i)
                .unwrap_or_else(|e| panic!("Failed to build BurstFrame for {}: {}", path.display(), e))
    })
    .collect();

    // Sanity check: all frames should have the same dimensions
    let (ref_h, ref_w) = (frames[0].height, frames[0].width);
    for f in &frames[1..] {
        assert_eq!(
            (f.height, f.width), (ref_h, ref_w),
            "Frame {} has different dimmensions than the reference!", f.frame_index
        );
    }

    println!(
        "Reference frame: {}x{}, ISO {:?}, {:?}s exposure",
        ref_w, ref_h, frames[0].iso, frames[0].exposure_time
    );

    println!("Aligning {} alternate frames...", frames.len() - 1);
    let alignments = align::align_burst(&frames);
    for (i, result) in alignments.iter().enumerate() {
        let center_tr = result.n_tile_rows / 2;
        let center_tc = result.n_tile_cols / 2;
        let (dr, dc) = result.offset_at(center_tr, center_tc);
        println!("  Frame {}: center tile offset = ({}, {})", i + 1, dr, dc);
    }

    println!("Merging...");
    let t0 = std::time::Instant::now();
    let merged = merge::merge_burst(&frames, &alignments);
    println!("Merge done ({:1}s)", t0.elapsed().as_secs_f32());
    println!("merged mean:   {:.6}", merged.mean().unwrap_or(0.0));

    // Save the merged Bayer as a grayscale PNG for now
    save_bayer_plane_png_raw(&merged, "merged_bayer.png");

    println!("Demosaicing...");
    let rgb = demosaic(&merged, frames[0].cfa_pattern);
    println!("rgb mean:      {:.6}", rgb.mean().unwrap_or(0.0));

    println!("wb_coeffs:     {:?}", frames[0].wb_coeffs);
    println!("color_matrix:  {:?}", frames[0].color_matrix);

    let finished = color_correction::finish(
        &rgb, 
        &frames[0].wb_coeffs, 
        &frames[0].color_matrix
    );
    println!("finished mean: {:.6}", finished.mean().unwrap_or(0.0));

    save_rgb_png(&finished, "output.png");
}

fn save_bayer_plane_png_raw(bayer: &ndarray::Array2<f32>, path: &str) {
    let (h, w) = bayer.dim();
    let pixels: Vec<u8> = bayer.iter()
        .map(|&v| {
            // Apply sRGB gamma (for debug display only)
            let gamma_corrected = v.clamp(0.0, 1.0).powf(1.0 / 2.2);
            (gamma_corrected * 255.0) as u8
        })
        .collect();
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::L8)
        .expect("Failed to save PNG");
}

fn save_rgb_png(rgb: &ndarray::Array3<f32>, path: &str) {
    let (h, w, _) = rgb.dim();
    let pixels: Vec<u8> = (0..h).flat_map(|r| {
        (0..w).flat_map(move |c| {
            (0..3).map(move |ch| {
                (rgb[[r, c, ch]].clamp(0.0, 1.0) * 255.0) as u8
            })
        })
    }).collect();
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)
        .expect("Failed to save PNG");
}
