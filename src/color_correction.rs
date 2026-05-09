use ndarray::Array3;

/// Full finishing pipeline: white balance + color correction + tone mapping + gamma.
/// Input: linear RGB Array3<f32> (h, w, 3), values between 0 and some float.
/// Output: sRGB Array3<f32> (h, w, 3), values between 0 and 1, ready to encode as PNG.
pub fn finish(
    rgb: &Array3<f32>, 
    wb_coeffs: &[f32; 4], 
    xyz_to_cam: &[[f32; 3]; 3]
) -> Array3<f32> {
    let (h, w, _) = rgb.dim();
    let mut out = Array3::<f32>::zeros((h, w, 3));

    let cam_to_srgb = compute_cam_to_srgb(xyz_to_cam);
    let wb = [
        wb_coeffs[0],
        (wb_coeffs[1] + wb_coeffs[2]) / 2.0, // Average Gr and Gb coeffs 
        wb_coeffs[3],
    ];

    // Normalize wb so green = 1.0 (keeps overall brightness stable)
    let wb = [wb[0] / wb[1], 1.0, wb[2] / wb[1]];

    for r in 0..h {
        for c in 0..w {
            // White balance
            let r_wb = rgb[[r, c, 0]] * wb[0];
            let g_wb = rgb[[r, c, 1]] * wb[1];
            let b_wb = rgb[[r, c, 2]] * wb[2];

            // Color correction (cam + sRGB via matrix multiply)
            let r_cc = cam_to_srgb[0][0]*r_wb + cam_to_srgb[0][1]*g_wb + cam_to_srgb[0][2]*b_wb;
            let g_cc = cam_to_srgb[1][0]*r_wb + cam_to_srgb[1][1]*g_wb + cam_to_srgb[1][2]*b_wb;
            let b_cc = cam_to_srgb[2][0]*r_wb + cam_to_srgb[2][1]*g_wb + cam_to_srgb[2][2]*b_wb;

            // Clamp negatives that can appear after color matrix (out-of-gamut colors)
            let r_cc = r_cc.max(0.0);
            let g_cc = g_cc.max(0.0);
            let b_cc = b_cc.max(0.0);

            // Tone mapping
            let r_tm = tone_map(r_cc);
            let g_tm = tone_map(g_cc);
            let b_tm = tone_map(b_cc);

            // sRGB gamma
            out[[r, c, 0]] = srgb_gamma(r_tm);
            out[[r, c, 1]] = srgb_gamma(g_tm);
            out[[r, c, 2]] = srgb_gamma(b_tm);
        }
    }

    out
}

// Reinhard tone mapping operator.
// Maps [0, INF] to [0, 1). Keeps shadows linear, compresses highlights.
// shoulder controls where compression starts (0.18 = photographic middle grey).
fn tone_map(x: f32) -> f32 {
    const SHOULDER: f32 = 0.08;
    x / (x + SHOULDER)
}

/// Convert linear light to perceptually uniform display values
/// sRGB gamma encoding (IEC 61966-2-1).
fn srgb_gamma(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Compute the combined cam+sRGB color matrix.
/// xyz_to_cam: 3*3 matrix converting CIE XYZ + camera native RGB
/// We need the reverse: camera native RGB -> sRGB.
/// Step 1: invert xyz_to_cam to get cam+XYZ
/// Step 2: multiply by the standard XYZ+sRGB matrix
fn compute_cam_to_srgb(xyz_to_cam: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    // Standard XYZ + sRGB matrix (IEC 61966-2-1, D65 white point)
    let xyz_to_srgb = [
        [ 3.2406f32, -1.5372, -0.4986],
        [-0.9689f32,  1.8758,  0.0415],
        [ 0.0557f32, -0.2040,  1.0570],
    ]; 

    // Invert xyz_to_cam using Cramer's rule (exact for 3*3)
    let cam_to_xyz = invert_3x3(xyz_to_cam);

    // Combined matrix: XYZ+sRGB * cam+XYZ
    mat_mul_3x3(&xyz_to_srgb, &cam_to_xyz)
}

/// 3*3 matrix inversion via Cramer's rule
fn invert_3x3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1]*m[2][2] - m[1][2]*m[2][1])
            - m[0][1] * (m[1][0]*m[2][2] - m[1][2]*m[2][0])
            + m[0][2] * (m[1][0]*m[2][1] - m[1][1]*m[2][0]);

    assert!(det.abs() > 1e-10, "Color matrix is singular, cannot invert");

    let inv_det = 1.0 / det;

    [
        [
            (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * inv_det,
            (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv_det,
            (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv_det,
        ],
        [
            (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * inv_det,
            (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv_det,
            (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv_det,
        ],
        [
            (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * inv_det,
            (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv_det,
            (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv_det,
        ],
    ]
}

/// 3*3 matrix multiplication
fn mat_mul_3x3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}
