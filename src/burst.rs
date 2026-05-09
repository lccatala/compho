use rawloader;
use ndarray::Array2;
use crate::metadata::read_dng_metadata;

#[derive(Debug, Clone)]
pub struct BurstFrame {
    // Core data
    pub bayer: Array2<f32>, // Linearized, black-level subtracted
    pub width: usize,
    pub height: usize,
    pub cfa_pattern: CfaPattern, // RGGB, BGGR, etc.

    // Sensor calibration
    pub white_level: f32, // Max output count for ADC before saturating to white
    pub black_level: f32, // Needed to do the subtraction at load-time
    pub wb_coeffs: [f32; 4], // From EXIF, auto white balance computed multipliers for [R, Gr, Gb, B]
    /// TODO: XYZ→camera matrix (NOT cam→XYZ, invert before use in finish.rs).
    /// Source: rawloader's `xyz_to_cam` field, rows 0..2 (row 3 is dropped padding).
    pub color_matrix: [[f32; 3]; 3], // To convert from camera's native RGB to XYZ (camera independent)

    pub noise_model: NoiseModel, // Needed for stage 3 merge
    
    // Burst metadata
    pub frame_index: usize, // Position in the burst (0 = reference)
    pub exposure_time: Option<f32>, // In seconds, from EXIF
    pub iso: Option<u32>, // ISO sensitivity, from EXIF
}

impl BurstFrame {
    /// Construct from a rawloader::RawImage
    /// 'frame_index' is 0 for the reference frame, 1..N for alternates
    pub fn from_raw_image(raw: &rawloader::RawImage, path: &str, frame_index: usize) -> Result<Self, String> {
        let u16_data = match &raw.data {
            rawloader::RawImageData::Integer(v) => v,
            rawloader::RawImageData::Float(_) => {
                return Err("Float raw data not supported".into());
            }
        };

        let dng = read_dng_metadata(path);

        let wb_coeffs = dng.wb_coeffs.unwrap_or_else(|| {
            // Rawloader fallback
            let raw_wb = raw.wb_coeffs;
            if raw_wb.iter().any(|x| x.is_nan() || *x <= 0.0) {
                [2.0, 1.0, 1.0, 1.5] // Neutral daylight estimate
            } else {
                raw_wb
            }
        });

        let color_matrix = dng.color_matrix.unwrap_or_else(|| {
            let xyz = raw.xyz_to_cam;
            [xyz[0], xyz[1], xyz[2]]
        });

        let exposure_time = dng.exposure_time;
        let iso = dng.iso;

        // black_level and white_level: rawloader gives per-channel arrays
        // but for a first pass we take the R-channel value for both
        // (they're almost always identical across channels)
        let black = raw.blacklevels[0] as f32;
        let white = raw.whitelevels[0] as f32;

        // Normalization: subtract black, divide by (white - black), clamp to [0,1]
        let range = white - black;
        let normalized: Vec<f32> = u16_data
            .iter()
            .map(|&px| ((px as f32 - black) / range).clamp(0.0, 1.0))
            .collect();

        // rawloader gives a flat Vec, reshape into (height * width) 2D array
        let bayer = Array2::from_shape_vec(
            (raw.height, raw.width), 
            normalized
        ).map_err(|e| e .to_string())?;

        // Parse CFA pattern from rawloader's string representation ("RGGB", etc.)
        let cfa_pattern = match raw.cfa.name.as_str() {
            "RGGB" => CfaPattern::RGGB,
            "BGGR" => CfaPattern::BGGR,
            "GRBG" => CfaPattern::GRBG,
            "GBRG" => CfaPattern::GBRG,
            other => return Err(format!("Unsupported CFA pattern {}", other)),
        };

        let noise_model = NoiseModel::default_estimate();

        Ok(BurstFrame {
            bayer,
            width: raw.width,
            height: raw.height,
            cfa_pattern,
            white_level: white,
            black_level: black,
            wb_coeffs: wb_coeffs,
            color_matrix,
            noise_model,
            frame_index,
            exposure_time,
            iso,
        })
    }
}

#[derive(Debug, Clone)]
pub struct NoiseModel {
    // Shot noise coefficient (signal-dependent variance)
    // VAriance from photon counting noise: o**2_shot = lambda_s * I
    pub lambda_s: f32,

    // Read noise coefficient (signal-independent floor)
    // Variance from electronics: o**2_read = lambda_r
    pub lambda_r: f32,
}

impl NoiseModel {
    /// Returns variance at a given normalized intensity level
    pub fn variance_at(&self, intensity: f32) -> f32 {
        self.lambda_s * intensity + self.lambda_r
    }

    /// Conservative defaults used when the RAW file has no noise metadata.
    /// Reasonable ballpark for a modern smartphone sensor at base ISO.
    pub fn default_estimate() -> Self {
        NoiseModel {
            lambda_s: 1e-4,
            lambda_r: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CfaPattern { RGGB, BGGR, GRBG, GBRG }

impl CfaPattern {
    /// Returns (row_offset, col_offset) within the 2x2 Bayer block for each channel
    /// These are the offsets you pass to ndarray's s![row_off..;2, col_off..;2]
    pub fn offsets(&self) -> ChannelOffsets {
        // The 2x2 block is laid out as
        //  (0,0) (0,1)
        //  (1,0) (1,1)
        // RGGB means: R at (0,0), Gr at (0,1), Gb at (1,0), B at (1,1)
        match self {
            CfaPattern::RGGB => ChannelOffsets { r: (0,0), gr: (0,1), gb: (1,0), b: (1,1) },
            CfaPattern::BGGR => ChannelOffsets { r: (1,1), gr: (1,0), gb: (0,1), b: (0,0) },
            CfaPattern::GRBG => ChannelOffsets { r: (0,1), gr: (0,0), gb: (1,1), b: (1,0) },
            CfaPattern::GBRG => ChannelOffsets { r: (1,0), gr: (1,1), gb: (0,0), b: (0,1) },
        }
    }
}

pub struct ChannelOffsets {
    pub r: (usize, usize),
    pub gr: (usize, usize),
    pub gb: (usize, usize),
    pub b: (usize, usize),
}
