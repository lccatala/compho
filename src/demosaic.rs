use ndarray::{Array3, Array2};
use crate::burst::CfaPattern;

/// Demosaics a Bayer array into a 3-channel RGB Array3<f32>.
/// Output shape: (height, width, 3), with channels in R, G, B order.
/// Values remain in linear light [0, 1], gamma is applied later.
pub fn demosaic(bayer: &Array2<f32>, pattern: CfaPattern) -> Array3<f32> {
    let (h, w) = bayer.dim();
    let offsets = pattern.offsets();

    // We work with the canonical RGGB layout internally.
    // If the pattern is different, 
    // we remap the output channels at the end rather than duplicating all the kernel logic.
    let mut rgb = Array3::<f32>::zeros((h, w, 3));

    // Copy known channel values directly
    // Each pixel already knows its own channel exactly, so no interpolation needed.
    // We use strided slices to address each Bayer plane.
    let (rr, rc) = offsets.r;
    let (grr, grc) = offsets.gr;
    let (gbr, gbc) = offsets.gb;
    let (br, bc) = offsets.b;

    // R pixels + channel 0
    for r in (rr..h).step_by(2) {
        for c in (rc..w).step_by(2) {
            rgb[[r, c, 0]] = bayer[[r, c]];
        }
    }

    // Gr pixels + channel 1
    for r in (grr..h).step_by(2) {
        for c in (grc..w).step_by(2) {
            rgb[[r, c, 1]] = bayer[[r, c]];
        }
    }

    // Gb pixels + channel 1 (both greens go to the same output channel)
    for r in (gbr..h).step_by(2) {
        for c in (gbc..w).step_by(2) {
            rgb[[r, c, 1]] = bayer[[r, c]];
        }
    }

    // B pixels + channel 2
    for r in (br..h).step_by(2) {
        for c in (bc..w).step_by(2) {
            rgb[[r, c, 2]] = bayer[[r, c]];
        }
    }

    // Interpolate missing channels with MHC kernels
    // Process each Bayer position type separately, using a 5*5 neighbourhood.
    // We skip a 2-pixel border to avoid bound checks inside the kernel loops.
    for r in 2..h-2 {
        for c in 2..w-2 {
            // Determine which Bayer position this pixel occupies
            let is_r  = (r % 2 == rr)  && (c % 2 == rc);
            let is_gr = (r % 2 == grr) && (c % 2 == grc);
            let is_gb = (r % 2 == gbr) && (c % 2 == gbc);
            let is_b  = (r % 2 == br)  && (c % 2 == bc);

            if is_r {
                // At R pixel: interpolate G and B
                rgb[[r, c, 1]] = interp_g_at_rb(bayer, r, c);
                rgb[[r, c, 2]] = interp_b_at_r(bayer, r, c);
            } else if is_b {
                // At B pixel: interpolate G and R
                rgb[[r, c, 1]] = interp_g_at_rb(bayer, r, c);
                rgb[[r, c, 0]] = interp_r_at_b(bayer, r, c);
            } else if is_gr {
                // At Gr pixel: interpolate R and B
                rgb[[r, c, 0]] = interp_r_at_gr(bayer, r, c);
                rgb[[r, c, 2]] = interp_b_at_r(bayer, r, c);
            } else if is_gb {
                // At Gb pixel: interpolate R and B
                rgb[[r, c, 0]] = interp_b_at_r(bayer, r, c); // transposed of Gb->B
                rgb[[r, c, 2]] = interp_b_at_gb(bayer, r, c);
            }
        }
    }

    // Fill 2-pixel border with bilinear (simple, barely visible)
    fill_border_bilinear(&mut rgb, bayer, &offsets, h, w);

    rgb
}

/// Estimate green at an R or B pixel.
/// Kernel: [0,0,-1,0,0 / 0,0,2,0,0 / -1,2,4,2,-1, / 0,0,2,0,0, / 0,0,-1,0,0] / 8
fn interp_g_at_rb(bayer: &Array2<f32>, r: usize, c: usize) -> f32 {
    (    -bayer[[r-2, c   ]]
         -bayer[[r  , c-2 ]]
  + 2.0 * bayer[[r-1, c   ]]
  + 2.0 * bayer[[r,   c-1 ]]
  + 4.0 * bayer[[r,   c   ]]
  + 2.0 * bayer[[r+1, c   ]]
  + 2.0 * bayer[[r,   c+1 ]]
         -bayer[[r+2, c   ]]
         -bayer[[r  , c+2 ]]
  ) / 8.0
}

/// Estimate R at a B pixel (or B at R, same kernel by symmetry).
/// Kernel: [0,0,-3/2,0,0 / 0,2,0,2,0 / -3,2,0,6,0,-3/2 / 0,2,0,2,0 / 0,0,-3/2,0,0] / 8
fn interp_b_at_r(bayer: &Array2<f32>, r: usize, c: usize) -> f32 {
    (  -1.5*bayer[[r-2, c  ]]
       -1.5*bayer[[r  , c-2]]
      + 2.0*bayer[[r-1, c-1]]
      + 2.0*bayer[[r-1, c+1]]
      + 6.0*bayer[[r  , c  ]]
      + 2.0*bayer[[r+1, c-1]]
      + 2.0*bayer[[r+1, c+1]]
       -1.5*bayer[[r+2, c  ]]
       -1.5*bayer[[r  , c+2]]
    ) / 8.0
}

fn interp_r_at_b(bayer: &Array2<f32>, r: usize, c: usize) -> f32 {
    interp_b_at_r(bayer, r, c)
}

/// Estimate R at Gr pixel.
/// Kernel: [0,0,1/2,0,0 / 0,-1,0,-1,0 / -1,4,5,4,-1 / 0,-1,0,-1,0 / 0,0,1/2,0,0] / 8
fn interp_r_at_gr(bayer: &Array2<f32>, r: usize, c: usize) -> f32 {
    (  0.5*bayer[[r-2, c  ]]
          -bayer[[r-1, c-1]]
          -bayer[[r-1, c+1]]
          -bayer[[r  , c-2]]
     + 4.0*bayer[[r  , c-1]]
     + 5.0*bayer[[r  , c  ]]
     + 4.0*bayer[[r  , c+1]]
          -bayer[[r  , c+2]]
          -bayer[[r+1, c-1]]
          -bayer[[r+1, c+1]]
     + 0.5*bayer[[r+2, c  ]]
    ) / 8.0
}

/// Estimate B at Gb pixel.
/// Same kernel as R-at-Gr but transposed.
fn interp_b_at_gb(bayer: &Array2<f32>, r: usize, c: usize) -> f32 {
    (0.5*bayer[[r  , c-2]]
        -bayer[[r-1, c-1]]
        -bayer[[r+1, c-1]]
        -bayer[[r-2, c  ]]
   + 4.0*bayer[[r-1, c  ]]
   + 5.0*bayer[[r  , c  ]]
   + 4.0*bayer[[r+1, c  ]]
        -bayer[[r+2, c  ]]
        -bayer[[r-1, c+1]]
        -bayer[[r+1, c+1]]
   + 0.5*bayer[[r  , c+2]]
    ) / 8.0
}

/// Fills the 2-pixel border with simple bilinear interpolation.
/// The border is a tiny fraction of the image and never appears in crops.
fn fill_border_bilinear(
    rgb: &mut Array3<f32>,
    bayer: &Array2<f32>,
    offsets: &crate::burst::ChannelOffsets,
    h: usize,
    w: usize,
) {
    // For each border pixel, average available same-channel neighbours.
    // Simplified fallback, correctness over quality
    for r in 0..h {
        for c in 0..w {
            // Only process border pixels
            if r >= 2 && r < h-2 && c >= 2 && c < w-2 {
                continue
            }

            let (rr, rc) = offsets.r;
            let chan = if (r % 2 == rr) && (c % 2 == rc) { 0 }
                else if (r % 2 == offsets.b.0) && (c % 2 == offsets.b.1) { 2 }
                else { 1 };

            // Known channel, just copy
            rgb[[r, c, chan]] = bayer[[r, c]];

            // Missing channels: clamp and average nearest same-channel pixels
            for target_chan in 0..3usize {
                if target_chan == chan {
                    continue;
                }

                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                for dr in -2i32..=2 {
                    for dc in -2i32..=2 {
                        let nr = (r as i32 + dr).clamp(0, h as i32 - 1) as usize;
                        let nc = (c as i32 + dc).clamp(0, w as i32 - 1) as usize;
                        let nc_chan = if (nr % 2 == rr) && (nc % 2 == rc) { 0 }
                            else if (nr % 2 == offsets.b.0) && (nc % 2 == offsets.b.1) { 2 }
                            else { 1 };
                        if nc_chan == target_chan {
                            sum += bayer[[nr, nc]];
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    rgb[[r, c, target_chan]] = sum / count;
                }
            }
        }
    }
}
