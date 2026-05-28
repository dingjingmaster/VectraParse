use std::fs;
use std::io::Cursor;
use std::path::Path;

use image::imageops::FilterType;
use image::DynamicImage;
use tract_onnx::prelude::*;

const EMBED_DET_ONNX: &[u8] = include_bytes!("../../../data/det.onnx");
const EMBED_REC_ZH_ONNX: &[u8] = include_bytes!("../../../data/chinese/rec.onnx");
const EMBED_REC_EN_ONNX: &[u8] = include_bytes!("../../../data/english/rec.onnx");
const EMBED_DICT_ZH: &str = include_str!("../../../data/chinese/dict.txt");
const EMBED_DICT_EN: &str = include_str!("../../../data/english/dict.txt");

#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub det_model_path: Option<String>,
    pub rec_model_path: Option<String>,
    pub rec_dict_path: Option<String>,
    pub rec_img_h: usize,
    pub rec_img_w: usize,
    pub rec_alt_model_path: Option<String>,
    pub rec_alt_dict_path: Option<String>,
    pub det_img_side: usize,
    pub det_box_thresh: f32,
    pub det_min_box_area: usize,
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub warning: Option<String>,
}

pub struct TractOcrEngine {
    det: TypedRunnableModel<TypedModel>,
    rec: TypedRunnableModel<TypedModel>,
    rec_alt: Option<TypedRunnableModel<TypedModel>>,
    alphabet: Vec<String>,
    alphabet_alt: Vec<String>,
}

impl TractOcrEngine {
    pub fn load(cfg: &OcrConfig) -> TractResult<Self> {
        let det = load_model(cfg.det_model_path.as_deref(), EMBED_DET_ONNX)?;
        let rec = load_model(cfg.rec_model_path.as_deref(), EMBED_REC_ZH_ONNX)?;
        let rec_alt = cfg
            .rec_alt_model_path
            .as_deref()
            .and_then(|p| {
                load_model(Some(p), EMBED_REC_EN_ONNX).ok()
            });
        let alphabet = load_dict(cfg.rec_dict_path.as_deref(), EMBED_DICT_ZH);
        let alphabet_alt = load_dict(cfg.rec_alt_dict_path.as_deref(), EMBED_DICT_EN);
        Ok(Self {
            det,
            rec,
            rec_alt,
            alphabet,
            alphabet_alt,
        })
    }

    pub fn infer(&self, image_bytes: &[u8], cfg: &OcrConfig) -> TractResult<OcrResult> {
        let img = image::load_from_memory(image_bytes)?;
        self.infer_image(&img, cfg)
    }

    fn infer_image(&self, img: &DynamicImage, cfg: &OcrConfig) -> TractResult<OcrResult> {
        let boxes = self.detect_text_boxes(img, cfg)?;
        let mut lines: Vec<(u32, u32, String, f32)> = Vec::new();
        for b in boxes {
            let crop = crop_box(img, b);
            let rec_input = preprocess_rec_image(&crop, cfg.rec_img_h, cfg.rec_img_w)?;
            let output = self.rec.run(tvec!(rec_input.into()))?;
            let logits = output[0].to_array_view::<f32>()?;
            let (text, confidence) = ctc_greedy_decode(&logits, &self.alphabet);
            if !text.trim().is_empty() {
                lines.push((b.1, b.0, text, confidence));
            }
        }
        lines.sort_by_key(|(y, x, _, _)| (*y / 8, *x));
        let mut text = lines
            .iter()
            .map(|(_, _, t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let mut confidence = if lines.is_empty() {
            0.0
        } else {
            lines.iter().map(|(_, _, _, c)| *c).sum::<f32>() / lines.len() as f32
        };
        if text.trim().is_empty() {
            let rec_input = preprocess_rec_image(img, cfg.rec_img_h, cfg.rec_img_w)?;
            let output = self.rec.run(tvec!(rec_input.into()))?;
            let logits = output[0].to_array_view::<f32>()?;
            let (fallback_text, fallback_confidence) = ctc_greedy_decode(&logits, &self.alphabet);
            text = fallback_text;
            confidence = fallback_confidence;
            if text.trim().is_empty()
                && let Some(rec_alt) = &self.rec_alt
            {
                let rec_input = preprocess_rec_image(img, cfg.rec_img_h, cfg.rec_img_w)?;
                let output = rec_alt.run(tvec!(rec_input.into()))?;
                let logits = output[0].to_array_view::<f32>()?;
                let (alt_text, alt_confidence) = ctc_greedy_decode(&logits, &self.alphabet_alt);
                text = alt_text;
                confidence = alt_confidence;
            }
        }
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
            det_model_path: None,
            rec_model_path: None,
            rec_dict_path: None,
            rec_img_h: 48,
            rec_img_w: 320,
            rec_alt_model_path: Some("data/english/rec.onnx".to_string()),
            rec_alt_dict_path: Some("data/english/dict.txt".to_string()),
            det_img_side: 960,
            det_box_thresh: 0.18,
            det_min_box_area: 20,
        }
    }
}

fn load_model(path: Option<&str>, embedded: &[u8]) -> TractResult<TypedRunnableModel<TypedModel>> {
    let model = if let Some(p) = path {
        tract_onnx::onnx().model_for_path(Path::new(p))?
    } else {
        let mut cursor = Cursor::new(embedded);
        tract_onnx::onnx().model_for_read(&mut cursor)?
    };
    model.into_optimized()?.into_runnable()
}

type BoxRect = (u32, u32, u32, u32);

impl TractOcrEngine {
    fn detect_text_boxes(&self, img: &DynamicImage, cfg: &OcrConfig) -> TractResult<Vec<BoxRect>> {
        let (det_input, sx, sy, w, h) = preprocess_det_image(img, cfg.det_img_side)?;
        let output = self.det.run(tvec!(det_input.into()))?;
        let map = output[0].to_array_view::<f32>()?;
        let mut boxes = extract_boxes_from_map(&map, cfg.det_box_thresh, cfg.det_min_box_area);
        for b in &mut boxes {
            b.0 = ((b.0 as f32) * sx).round() as u32;
            b.1 = ((b.1 as f32) * sy).round() as u32;
            b.2 = ((b.2 as f32) * sx).round() as u32;
            b.3 = ((b.3 as f32) * sy).round() as u32;
            b.0 = b.0.min(w.saturating_sub(1));
            b.1 = b.1.min(h.saturating_sub(1));
            b.2 = b.2.min(w);
            b.3 = b.3.min(h);
        }
        boxes.retain(|(x0, y0, x1, y1)| x1 > x0 && y1 > y0);
        if boxes.is_empty() {
            boxes.push((0, 0, w, h));
        }
        Ok(boxes)
    }
}

fn load_dict(path: Option<&str>, embedded: &str) -> Vec<String> {
    let content = if let Some(p) = path {
        fs::read_to_string(p).unwrap_or_else(|_| embedded.to_string())
    } else {
        embedded.to_string()
    };
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn preprocess_det_image(
    image: &DynamicImage,
    side: usize,
) -> TractResult<(Tensor, f32, f32, u32, u32)> {
    let rgb = image.to_rgb8();
    let (src_w, src_h) = rgb.dimensions();
    let resized = image::imageops::resize(&rgb, side as u32, side as u32, FilterType::Triangle);
    let mut data = vec![0f32; 1 * 3 * side * side];
    for y in 0..side {
        for x in 0..side {
            let px = resized.get_pixel(x as u32, y as u32);
            let bgr = [px[2] as f32, px[1] as f32, px[0] as f32];
            let norm = [
                ((bgr[0] / 255.0) - 0.485) / 0.229,
                ((bgr[1] / 255.0) - 0.456) / 0.224,
                ((bgr[2] / 255.0) - 0.406) / 0.225,
            ];
            for c in 0..3 {
                let idx = c * side * side + y * side + x;
                data[idx] = norm[c];
            }
        }
    }
    let arr = tract_ndarray::Array4::from_shape_vec((1, 3, side, side), data)?;
    let sx = src_w as f32 / side as f32;
    let sy = src_h as f32 / side as f32;
    Ok((arr.into_tensor(), sx, sy, src_w, src_h))
}

fn extract_boxes_from_map(
    map: &tract_ndarray::ArrayViewD<'_, f32>,
    thresh: f32,
    min_area: usize,
) -> Vec<BoxRect> {
    if map.ndim() != 4 {
        return Vec::new();
    }
    let h = map.shape()[2];
    let w = map.shape()[3];
    let mut mask = vec![false; h * w];
    for y in 0..h {
        for x in 0..w {
            let v = map[[0, 0, y, x]];
            mask[y * w + x] = v >= thresh;
        }
    }
    let mut visited = vec![false; h * w];
    let mut boxes = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || !mask[idx] {
                continue;
            }
            let mut queue = vec![(x, y)];
            visited[idx] = true;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut area = 0usize;
            while let Some((cx, cy)) = queue.pop() {
                area += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                let neigh = [
                    (cx.wrapping_sub(1), cy),
                    (cx + 1, cy),
                    (cx, cy.wrapping_sub(1)),
                    (cx, cy + 1),
                ];
                for (nx, ny) in neigh {
                    if nx >= w || ny >= h {
                        continue;
                    }
                    let nidx = ny * w + nx;
                    if visited[nidx] || !mask[nidx] {
                        continue;
                    }
                    visited[nidx] = true;
                    queue.push((nx, ny));
                }
            }
            if area >= min_area {
                boxes.push((min_x as u32, min_y as u32, (max_x + 1) as u32, (max_y + 1) as u32));
            }
        }
    }
    boxes
}

fn crop_box(img: &DynamicImage, b: BoxRect) -> DynamicImage {
    let (x0, y0, x1, y1) = b;
    let w = x1.saturating_sub(x0).max(1);
    let h = y1.saturating_sub(y0).max(1);
    img.crop_imm(x0, y0, w, h)
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
            let bgr = [px[2], px[1], px[0]];
            for c in 0..3 {
                let v = (bgr[c] as f32 / 255.0 - 0.5) / 0.5;
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
    let (steps, classes, channel_first) = if shape[1] > shape[2] {
        (shape[2], shape[1], true)
    } else {
        (shape[1], shape[2], false)
    };
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
            let v = if channel_first {
                logits[[0, c, t]]
            } else {
                logits[[0, t, c]]
            };
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
