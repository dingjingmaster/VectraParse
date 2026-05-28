use std::fs;
use std::path::Path;

use image::imageops::FilterType;
use image::DynamicImage;
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub det_model_path: String,
    pub rec_model_path: String,
    pub rec_dict_path: Option<String>,
    pub rec_img_h: usize,
    pub rec_img_w: usize,
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub warning: Option<String>,
}

pub struct TractOcrEngine {
    rec: TypedRunnableModel<TypedModel>,
    alphabet: Vec<String>,
}

impl TractOcrEngine {
    pub fn load(cfg: &OcrConfig) -> TractResult<Self> {
        // Keep loading det model to validate model presence for future integration.
        let _ = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.det_model_path))?
            .into_optimized()?;
        let rec = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.rec_model_path))?
            .into_optimized()?
            .into_runnable()?;
        let alphabet = load_dict(cfg.rec_dict_path.as_deref());
        Ok(Self { rec, alphabet })
    }

    pub fn infer(&self, image_bytes: &[u8], cfg: &OcrConfig) -> TractResult<OcrResult> {
        let img = image::load_from_memory(image_bytes)?;
        self.infer_image(&img, cfg)
    }

    fn infer_image(&self, img: &DynamicImage, cfg: &OcrConfig) -> TractResult<OcrResult> {
        let rec_input = preprocess_rec_image(img, cfg.rec_img_h, cfg.rec_img_w)?;
        let output = self.rec.run(tvec!(rec_input.into()))?;
        let logits = output[0].to_array_view::<f32>()?;
        let (text, confidence) = ctc_greedy_decode(&logits, &self.alphabet);
        let warning = if self.alphabet.is_empty() {
            Some("ocr-dictionary-missing".to_string())
        } else {
            None
        };
        Ok(OcrResult {
            text,
            confidence,
            warning,
        })
    }
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            det_model_path: "data/ch_PP-OCRv4_det.onnx".to_string(),
            rec_model_path: "data/ch_PP-OCRv4_rec.onnx".to_string(),
            rec_dict_path: Some("data/ppocr_keys_v1.txt".to_string()),
            rec_img_h: 48,
            rec_img_w: 320,
        }
    }
}

fn load_dict(path: Option<&str>) -> Vec<String> {
    let Some(path) = path else {
        return Vec::new();
    };
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn preprocess_rec_image(image: &DynamicImage, target_h: usize, target_w: usize) -> TractResult<Tensor> {
    let rgb = image.to_rgb8();
    let (src_w, src_h) = rgb.dimensions();
    let ratio = src_w as f32 / src_h as f32;
    let mut resized_w = (ratio * target_h as f32).ceil() as usize;
    resized_w = resized_w.clamp(1, target_w);
    let resized = image::imageops::resize(
        &rgb,
        resized_w as u32,
        target_h as u32,
        FilterType::Triangle,
    );

    let mut data = vec![0f32; 1 * 3 * target_h * target_w];
    for y in 0..target_h {
        for x in 0..resized_w {
            let px = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = (px[c] as f32 / 255.0 - 0.5) / 0.5;
                let idx = c * target_h * target_w + y * target_w + x;
                data[idx] = v;
            }
        }
    }
    let arr = tract_ndarray::Array4::from_shape_vec((1, 3, target_h, target_w), data)?;
    Ok(arr.into_tensor())
}

fn ctc_greedy_decode(logits: &tract_ndarray::ArrayViewD<'_, f32>, alphabet: &[String]) -> (String, f32) {
    if logits.ndim() != 3 {
        return (String::new(), 0.0);
    }
    let shape = logits.shape();
    let steps = shape[1];
    let classes = shape[2];
    if classes <= 1 {
        return (String::new(), 0.0);
    }
    let blank_id = 0usize;
    let mut prev = blank_id;
    let mut text = String::new();
    let mut prob_sum = 0.0f32;
    let mut count = 0usize;

    for t in 0..steps {
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..classes {
            let v = logits[[0, t, c]];
            if v > best_val {
                best_val = v;
                best_id = c;
            }
        }
        if best_id != blank_id && best_id != prev {
            if let Some(ch) = alphabet.get(best_id - 1) {
                text.push_str(ch);
            } else {
                text.push('?');
            }
            prob_sum += best_val;
            count += 1;
        }
        prev = best_id;
    }
    let confidence = if count == 0 { 0.0 } else { prob_sum / count as f32 };
    (text, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_result_is_empty() {
        let out = OcrResult::default();
        assert!(out.text.is_empty());
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn default_config_points_to_ppocrv4_paths() {
        let cfg = OcrConfig::default();
        assert!(cfg.det_model_path.contains("PP-OCRv4_det.onnx"));
        assert!(cfg.rec_model_path.contains("PP-OCRv4_rec.onnx"));
    }
}
