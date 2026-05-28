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
    alphabet: Vec<String>,
}

impl TractOcrEngine {
    pub fn load(cfg: &OcrConfig) -> TractResult<Self> {
        let det = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.det_model_path))?
            .into_optimized()?
            .into_runnable()?;
        let rec = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.rec_model_path))?
            .into_optimized()?
            .into_runnable()?;
        let alphabet = load_dict(cfg.rec_dict_path.as_deref());
        Ok(Self { det, rec, alphabet })
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
        let text = lines
            .iter()
            .map(|(_, _, t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let confidence = if lines.is_empty() {
            0.0
        } else {
            lines.iter().map(|(_, _, _, c)| *c).sum::<f32>() / lines.len() as f32
        };
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
            rec_dict_path: None,
            rec_img_h: 48,
            rec_img_w: 320,
            det_img_side: 960,
            det_box_thresh: 0.3,
            det_min_box_area: 32,
        }
    }
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

fn load_dict(path: Option<&str>) -> Vec<String> {
    let candidates: Vec<&str> = match path {
        Some(p) => vec![p],
        None => vec![
            "data/chinese/dict.txt",
            "data/english/dict.txt",
            "data/ppocr_keys_v1.txt",
        ],
    };
    let mut content_opt = None;
    for p in candidates {
        if let Ok(content) = fs::read_to_string(p) {
            content_opt = Some(content);
            break;
        }
    }
    let Some(content) = content_opt else {
        return Vec::new();
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
            for c in 0..3 {
                let v = px[c] as f32 / 255.0;
                let idx = c * side * side + y * side + x;
                data[idx] = v;
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
