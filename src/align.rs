use ndarray::{Array2, ArrayView2, s};
use rayon::prelude::*;
use crate::burst::BurstFrame;

/// The alignment result for one alternate frame against the reference.
/// Stores a per-tile (row, col) pixel offset at full resolution.
pub struct AlignmentResult {
    pub offsets: Vec<(i32, i32)>, // flat: index = tile_row * n_tile_cols + tile_col
    pub n_tile_rows: usize,
    pub n_tile_cols: usize,
    pub tile_size: usize,
}

impl AlignmentResult {
    pub fn offset_at(&self, tile_row: usize, tile_col: usize) -> (i32, i32) {
        self.offsets[tile_row * self.n_tile_cols + tile_col]
    }
}

/// Builds a coarse-to-fine pyramid by blurring and subsampling.
/// Level 0 is the original. Each subsequent level is half the resolution.
pub fn build_pyramid(image: &Array2<f32>, levels: usize) -> Vec<Array2<f32>> {
    let mut pyramid = vec![image.clone()]; // Level 0: full resolution

    for _ in 1..levels {
        let prev = pyramid.last().unwrap(); // Take most recently added level
        let blurred = gaussian_blur_5x5(prev); // Blur it
        pyramid.push(subsample_by_2(&blurred)); // Subsample and add to pyramid
    }
    pyramid
}

/// Averages each 2x2 block into one output pixel.
/// On Bayer data this naturally combines all four color channels.
fn subsample_by_2(image: &Array2<f32>) -> Array2<f32> {
    let (h, w) = image.dim();
    Array2::from_shape_fn((h / 2, w / 2), |(r, c)| {
        (image[[2*r,   2*c  ]] +
         image[[2*r,   2*c+1]] +
         image[[2*r+1, 2*c  ]] +
         image[[2*r+1, 2*c+1]]) / 4.0
    })
}

/// Separable 5x5 gaussian blur, o=1. Applied before downsampling to prevent aliasing.
/// Kernel: [1, 4, 6, 4, 1] / 16 (applied horizontally then vertically)
fn gaussian_blur_5x5(image: &Array2<f32>) -> Array2<f32> {
    let kernel = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    let norm = 16.0f32;
    let (h, w) = image.dim();

    // Horizontal pass
    let mut horiz = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let ci = (c as i32 + k as i32 - 2).clamp(0, w as i32 - 1) as usize;
                acc += image[[r, ci]] * kv;
            }
            horiz[[r, c]] = acc / norm;
        }
    }

    // Vertical pass
    let mut out = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let ri = (r as i32 + k as i32 - 2).clamp(0, h as i32 - 1) as usize;
                acc += horiz[[ri, c]] * kv;
            }
            out[[r, c]] = acc / norm;
        }
    }
    out
}

/// Extracts a tile from 'image', centered at '(origin_r + offset.0, origin_c + offset.1)'
/// Clamps to image boundaries so you never need to bounds-check at the call site.
fn extract_tile(
    image: &Array2<f32>,
    tile_origin: (usize, usize),
    offset: (i32, i32),
    tile_size: usize
) -> Array2<f32> {
    let (h, w) = image.dim();
    let r0 = (tile_origin.0 as i32 + offset.0).clamp(0, h as i32 - tile_size as i32) as usize;
    let c0 = (tile_origin.1 as i32 + offset.1).clamp(0, w as i32 - tile_size as i32) as usize;
    image.slice(s![r0..r0+tile_size, c0..c0+tile_size]).to_owned()
}

/// L1 norm between two same-sized tiles. Robust to outliers (saturated pixels, dust).
fn l1_distance(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> f32 {
    ndarray::Zip::from(a)
        .and(b)
        .fold(0.0f32, |acc, &x, &y| acc + (x - y).abs())
}

/// Exhaustive search over a +-search_range window around initial_offset.
/// Returns the offset that minimizes L1 distance between the reference tile
/// and the corresponding region in 'alternate'.
fn best_offset(
    ref_tile: &ArrayView2<f32>,
    alternate: &Array2<f32>,
    tile_origin: (usize, usize),
    initial_offset: (i32, i32),
    search_range: i32,
    tile_size: usize
) -> (i32, i32) {
    let (h, w) = alternate.dim();
    let mut best_cost = f32::INFINITY;
    let mut best = initial_offset;

    for dr in -search_range..=search_range {
        for dc in -search_range..=search_range {
            let candidate = (initial_offset.0 + dr, initial_offset.1 + dc);
            let r0 = (tile_origin.0 as i32 + candidate.0)
                .clamp(0, h as i32 - tile_size as i32) as usize;
            let c0 = (tile_origin.1 as i32 + candidate.1)
                .clamp(0, w as i32 - tile_size as i32) as usize;
            let alt_view = alternate.slice(s![r0..r0+tile_size, c0..c0+tile_size]);
            let cost = l1_distance(ref_tile, &alt_view);
            if cost < best_cost {
                best_cost = cost;
                best = candidate;
            }
        }
    }
    best
}

const PYRAMID_LEVELS: usize = 4;
const TILE_SIZE: usize = 16; // At finest level, in pixels
const SEARCH_RANGE: i32 = 4; // +- 4 tiles at each pyramid level

/// Align one alternate frame against the reference pyramid.
pub fn align_frame(
    ref_pyramid: &[Array2<f32>],
    alt_pyramid: &[Array2<f32>],
) -> AlignmentResult {
    let (full_h, full_w) = ref_pyramid[0].dim();
    let n_tile_rows = full_h / TILE_SIZE;
    let n_tile_cols = full_w / TILE_SIZE;
    let n_tiles = n_tile_rows * n_tile_cols;

    // Start with zero offsets at the coarsest level
    let mut offsets: Vec<(i32, i32)> = vec![(0, 0); n_tiles];

    // Coarsest + finest (PYRAMID_LEVELS-1 down to 0)
    for level in (0..PYRAMID_LEVELS).rev() {
        let ref_level = &ref_pyramid[level];
        let alt_level = &alt_pyramid[level];
        let scale = 1usize << level; // How many full-res pixels per level pixel
        let tile_size_at_level = (TILE_SIZE / scale).max(1);

        let new_offsets: Vec<(i32, i32)> = (0..n_tiles)
            .into_par_iter()
            .map(|idx| {
                let tr = idx / n_tile_cols;
                let tc = idx % n_tile_cols;

                // Scale tile origin down to this pyramid level
                 let origin_r = (tr * TILE_SIZE) / scale;
                 let origin_c = (tc * TILE_SIZE) / scale;

                 let initial = if level == PYRAMID_LEVELS - 1 {
                     (0, 0)
                 } else {
                     let prev = offsets[idx];
                     (prev.0 * 2, prev.1 * 2)
                 };

                let ref_tile = ref_level.slice(s![origin_r..origin_r+tile_size_at_level,
                                   origin_c..origin_c+tile_size_at_level]);

                 best_offset(
                     &ref_tile,
                     alt_level,
                     (origin_r, origin_c),
                     initial,
                     SEARCH_RANGE,
                     tile_size_at_level,
                 )
            })
            .collect();
        offsets = new_offsets;
    }
    // Offsets are currently in level-0 (full-resolution) pixel coordinates
    AlignmentResult { offsets, n_tile_rows, n_tile_cols, tile_size: TILE_SIZE }
}

/// Align all alternate frames in the burst against frame 0 (the reference).
pub fn align_burst(frames: &[BurstFrame]) -> Vec<AlignmentResult> {
    let ref_pyramid = build_pyramid(&frames[0].bayer, PYRAMID_LEVELS);

    frames[1..].iter().enumerate().map(|(i, frame)| {
        let t0 = std::time::Instant::now();
        let alt_pyramid = build_pyramid(&frame.bayer, PYRAMID_LEVELS);
        print!(
            "  Frame {}/{}: pyramid built ({:.1}s), aligning...", 
            i+1, 
            frames.len() - 1, 
            t0.elapsed().as_secs_f32()
        );
        // flush so the partial line prints before alignment starts
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        let result = align_frame(&ref_pyramid, &alt_pyramid);
        let center = result.offset_at(result.n_tile_rows / 2, result.n_tile_cols / 2);
        println!(
            "done ({:.1}s - center tile offset: {:?})",
            t0.elapsed().as_secs_f32(), center
        );
        result
    }).collect()
}
