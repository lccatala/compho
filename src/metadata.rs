use exif::{Reader, Tag, In, Value, Field};
use std::io::BufReader;
use std::fs::File;

pub struct DngMetadata {
    /// White balance multipliers [R, Gr, Gb, B], normalized so G=1.
    /// Derived from the AsShotNeutral DNG tag.
    pub wb_coeffs: Option<[f32; 4]>,
    /// XYZ+camera color matrix (3*3), from ColorMatrix1 DNG tag.
    pub color_matrix: Option<[[f32; 3]; 3]>,
    /// Exposure time in seconds, from standard EXIF
    pub exposure_time: Option<f32>,
    /// ISO sensitivity, from standard EXIF
    pub iso: Option<u32>,
}

pub fn read_dng_metadata(path: &str) -> DngMetadata {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("metadata: could not open {}: {}", path, e);
            return DngMetadata { wb_coeffs: None, color_matrix: None, exposure_time: None, iso: None }
        }
    };

    let exif = match Reader::new().read_from_container(&mut BufReader::new(file)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("metadata: could not parse EXIF from {}: {}", path, e);
            return DngMetadata { wb_coeffs: None, color_matrix: None, exposure_time: None, iso: None }
        }
    };

    DngMetadata { 
        wb_coeffs:     read_as_shot_neutral(&exif),
        color_matrix:  read_color_matrix(&exif),
        exposure_time: read_exposure_time(&exif),
        iso:           read_iso(&exif),
    }
}

fn find_tag(exif: &exif::Exif, tag_number: u16) -> Option<&Field> {
    exif.fields().find(|f| f.tag.number() == tag_number)
}
fn read_as_shot_neutral(exif: &exif::Exif) -> Option<[f32; 4]> {
    let field = find_tag(exif, 0xC628)?; // AsShotNeutral

    let rationals = match &field.value {
        Value::Rational(v) => v,
        _ => return None,
    };

    if rationals.len() < 3 {
        return None;
    }

    let r_neutral = rationals[0].num as f32 / rationals[0].denom as f32;
    let g_neutral = rationals[1].num as f32 / rationals[1].denom as f32;
    let b_neutral = rationals[2].num as f32 / rationals[2].denom as f32;

    if g_neutral == 0.0 {
        return None;
    }

    // Convert neutral + multiplier, normalize as G multiplier = 1.0.
    // multiplier = 1/neutral; normalized = multiplier / g_multiplier = g_neutral / neutral
    let r_mult = g_neutral / r_neutral;
    let g_mult = 1.0_f32;
    let b_mult = g_neutral / b_neutral;

    Some([r_mult, g_mult, g_mult, b_mult])
}

fn read_color_matrix(exif: &exif::Exif) -> Option<[[f32; 3]; 3]> {
    let field = find_tag(exif, 0xC621)?; // ColorMatrix1

    let rationals = match &field.value {
        Value::SRational(v) => v,
        _ => return None,
    };

    if rationals.len() < 9 {
        return None;
    }

    let v: Vec<f32> = rationals.iter()
        .map(|r| r.num as f32 / r.denom as f32)
        .collect();

    Some([
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5]],
        [v[6], v[7], v[8]],
    ])
}

fn read_exposure_time(exif: &exif::Exif) -> Option<f32> {
    let field = exif.get_field(exif::Tag::ExposureTime, In::PRIMARY)?;
    match &field.value {
        Value::Rational(v) if !v.is_empty() =>
            Some(v[0].num as f32 / v[0].denom as f32),
        _ => None,
    }
}

fn read_iso(exif: &exif::Exif) -> Option<u32> {
    let field = exif.get_field(exif::Tag::PhotographicSensitivity, In::PRIMARY)
        .or_else(|| exif.get_field(exif::Tag::ISOSpeed, In::PRIMARY))?;
    match &field.value {
        Value::Short(v) if !v.is_empty() => Some(v[0] as u32),
        Value::Long(v) if !v.is_empty() => Some(v[0]),
        _ => None,
    }

}
