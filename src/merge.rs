use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

use ndarray::{Array2, s};
use ndrustfft::{Complex};
use rayon::prelude::*;
use crate::burst::BurstFrame;
use crate::align::{AlignmentResult};

// Tiles overlap by 50% to avoid blocking artifacts at boundaries.
// So the step between tile origins is TILE_SIZE / 2.
const OVERLAP: usize = 2; // divisor - step = tile_size / OVERLAP

/// Full merge pipeline for one burst.
/// Returns a single denoised Bayer Array2<f32>, same dimensions as the reference.
pub fn merge_burst(frames: &[BurstFrame], alignments: &[AlignmentResult]) -> Array2<f32> {
    assert_eq!(frames.len() -1, alignments.len());

    let reference = &frames[0].bayer;
    let (h, w) = reference.dim();
    let tile_size = alignments[0].tile_size;
    let step = tile_size / OVERLAP;

    // Accumulator and weights for overlap-add reconstruction
    let mut accum = Array2::<f32>::zeros((h, w));
    let mut weights = Array2::<f32>::zeros((h, w));

    // 2D Hann window (precomputed once, resued for every tile)
    let window = hann_window_2d(tile_size);

    // Iterate over all tile origins with 50% overlap
    let tile_origins: Vec<(usize, usize)> = (0..h - tile_size + 1)
        .step_by(step)
        .flat_map(|r| (0..w - tile_size + 1).step_by(step).map(move |c| (r, c)))
        .collect();


    let mut planner = FftPlanner::<f32>::new();
    let fft_forward: Arc<dyn Fft<f32>> = planner.plan_fft_forward(tile_size);
    let fft_inverse: Arc<dyn Fft<f32>> = planner.plan_fft_inverse(tile_size);

    // Process tiles in parallel. Each produces an independent (origin, merged_tile) pair
    let merged_tiles: Vec<((usize, usize), Array2<f32>)> = tile_origins
        .into_par_iter()
        .map(|(r0, c0)| {
            let merged = merge_tile(
                frames, 
                alignments, 
                &window, 
                r0, c0, tile_size,
                &fft_forward, &fft_inverse
            );
            ((r0, c0), merged)
        })
        .collect();

    // Overlap-add accumulation (single-threaded since it writes to shared arrays)
    for ((r0, c0), tile) in merged_tiles {
        for tr in 0..tile_size {
            for tc in 0..tile_size {
                accum[[r0+tr, c0+tc]] += tile[[tr, tc]] * window[[tr, tc]];
                weights[[r0+tr, c0+tc]] += window[[tr, tc]] * window[[tr, tc]];
            }
        }
    }
    
    // Normalize by accumulated window weights
    accum.zip_mut_with(&weights, |a, &wt| {
        if wt > 1e-8 { *a /= wt; }
    });

    accum
}

/// Merge one tile across all frames using the Wiener filter in the DFT domain
fn merge_tile(
    frames: &[BurstFrame], 
    alignments: &[AlignmentResult], 
    window: &Array2<f32>, 
    r0: usize, c0: usize, tile_size: usize,
    fft_forward: &Arc<dyn Fft<f32>>,
    fft_inverse: &Arc<dyn Fft<f32>>,
) -> Array2<f32> {
    let n_alternates = alignments.len();

    // Extract and window the reference tile
    let ref_tile = extract_windowed_tile(&frames[0].bayer, window, r0, c0, tile_size);

    // DFT the reference tile
    let ref_dft = dft_2d(&ref_tile, tile_size, fft_forward);

    // Compute noise power for this tile
    // Extimate average intensity of the tile, use noise model to get variance
    let mean_intensity = ref_tile.mean().unwrap_or(0.1);
    let noise_var = frames[0].noise_model.variance_at(mean_intensity);

    // Compute Wiener weights per frequency bin
    let wiener_weights: Vec<f32> = ref_dft.iter().map(|&r_hat| {
        let signal_power = r_hat.norm_sqr();
        signal_power / (signal_power + n_alternates as f32 * noise_var)
    }).collect();

    // Accumulate DFTs from aligned alternate tiles
    let mut dft_sum: Vec<Complex<f32>> = ref_dft.clone();
    for (alt_idx, alignment) in alignments.iter().enumerate() {
        // Find which tile cell this origin maps to
        let tile_col = c0 / tile_size; // approximate
        let tile_row = r0 / tile_size;
        let (dr, dc) = alignment.offset_at(
            tile_row.min(alignment.n_tile_rows - 1), 
            tile_col.min(alignment.n_tile_cols - 1), 
        );

        // Extract the alternate tile at the aligned position
        let alt_r0 = (r0 as i32 + dr).clamp(0, frames[alt_idx+1].bayer.nrows() as i32 - tile_size as i32) as usize;
        let alt_c0 = (c0 as i32 + dc).clamp(0, frames[alt_idx+1].bayer.ncols() as i32 - tile_size as i32) as usize;
        let alt_tile = extract_windowed_tile(&frames[alt_idx+1].bayer, window, alt_r0, alt_c0, tile_size);
        let alt_dft = dft_2d(&alt_tile, tile_size, fft_forward);
        for (sum, a) in dft_sum.iter_mut().zip(alt_dft.iter()) {
            *sum += a;
        }
    }

    // Apply Wiener weights and normalize
    let merged_dft: Vec<Complex<f32>> = dft_sum.iter()
        .zip(wiener_weights.iter())
        .map(|(&s, &w)| s * w / (1.0 + n_alternates as f32 * w))
        .collect();

    // DFT back to spatial domain
    idft_2d(&merged_dft, tile_size, fft_inverse)
}

/// Extract a tile from 'image' at (r0, c0) and multiply by the Hann window
fn extract_windowed_tile(
    image: &Array2<f32>,
    window: &Array2<f32>,
    r0: usize,
    c0: usize,
    tile_size: usize,
) -> Array2<f32> {
    let tile = image.slice(s![r0..r0+tile_size, c0..c0+tile_size]);
    &tile * window // element-wise multiply (ndarray operator overloading)
}

/// 2D Hann window of size n*n
/// Each element = hann(c), where hann(i) = 0.5*(1 - cos(2PI * i / (n-1)))
/// Tapers to 0 at the edges, peaks at 1 at the center
fn hann_window_2d(n: usize) -> Array2<f32> {
    let hann_1d: Vec<f32> = (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()))
        .collect();
    Array2::from_shape_fn((n, n), |(r, c)| hann_1d[r] * hann_1d[c])
}

/// 2D DFT via row-then-column 1D FFTs
/// Returns a flat Vec of complex values in row-major order, shape n*n
fn dft_2d(tile: &Array2<f32>, n: usize, fft: &Arc<dyn Fft<f32>>) -> Vec<Complex<f32>> {    
    // Convert to complex and do row FFTs
    let mut buffer: Vec<Vec<Complex<f32>>> = (0..n).map(|r| {
        let mut row: Vec<Complex<f32>> = (0..n)
            .map(|c| Complex {re: tile[[r, c]], im: 0.0})
            .collect();
        fft.process(&mut row);
        row
    }).collect();

    // Column FFTs in-place
    for c in 0..n {
        let mut col: Vec<Complex<f32>> = (0..n).map(|r| buffer[r][c]).collect();
        fft.process(&mut col);
        for r in 0..n { buffer[r][c] = col[r]; }
    }

    // Flatten to row-major Vec
    (0..n).flat_map(|r| buffer[r].iter().copied()).collect()
}

/// Inverse 2D DFT. Mirrors dft_2d but uses the inverse FFT
fn idft_2d(dft: &[Complex<f32>], n: usize, ifft: &Arc<dyn Fft<f32>>) -> Array2<f32> {
    let scale = 1.0 / (n * n) as f32;

    // Reshape flat Vec into 2D buffer
    let mut buffer: Vec<Vec<Complex<f32>>> = (0..n)
        .map(|r| dft[r*n..(r+1)*n].to_vec())
        .collect();

    // Column IFFTs first, then row IFFTs (mirrors the forward order reversed)
    for c in 0..n {
        let mut col: Vec<Complex<f32>> = (0..n).map(|r| buffer[r][c]).collect();
        ifft.process(&mut col);
        for r in 0..n { buffer[r][c] = col[r]; }
    }
    for r in 0..n { ifft.process(&mut buffer[r]); }

    // Take the real part and scale (rustfft doesn't normalize automatically)
    Array2::from_shape_fn((n, n), |(r, c)| buffer[r][c].re * scale)
}

