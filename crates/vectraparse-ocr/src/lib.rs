use std::path::Path;

use image::imageops::FilterType;
use image::DynamicImage;

mod ort;

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
            det_box_thresh: 0.25,
            det_min_box_area: 30,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub warning: Option<String>,
}

pub struct OrtOcrEngine {
    det: OrtSession,
    rec: OrtSession,
    rec_alt: Option<OrtSession>,
    alphabet: Vec<String>,
    alphabet_alt: Vec<String>,
}

struct OrtSession {
    session_ptr: *mut ort::OrtSession,
    allocator_ptr: *mut ort::OrtAllocator,
    memory_info_ptr: *mut ort::OrtMemoryInfo,
}

unsafe impl Send for OrtSession {}
unsafe impl Sync for OrtSession {}

impl Drop for OrtSession {
    fn drop(&mut self) {
        if !self.session_ptr.is_null() {
            ort::release_session(self.session_ptr);
        }
        if !self.allocator_ptr.is_null() {
            ort::release_allocator(self.allocator_ptr);
        }
        if !self.memory_info_ptr.is_null() {
            ort::release_memory_info(self.memory_info_ptr);
        }
    }
}

impl OrtOcrEngine {
    pub fn load(cfg: &OcrConfig) -> Result<Self, String> {
        ort::ensure_initialized()?;

        let det = load_ort_session(cfg.det_model_path.as_deref(), EMBED_DET_ONNX)
            .map_err(|e| format!("det model: {e}"))?;
        let rec = load_ort_session(cfg.rec_model_path.as_deref(), EMBED_REC_ZH_ONNX)
            .map_err(|e| format!("rec model: {e}"))?;
        let rec_alt = cfg
            .rec_alt_model_path
            .as_deref()
            .and_then(|p| load_ort_session(Some(p), EMBED_REC_EN_ONNX).ok());
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

    pub fn infer(&self, image_bytes: &[u8], cfg: &OcrConfig) -> Result<OcrResult, String> {
        let img = image::load_from_memory(image_bytes).map_err(|e| format!("image decode: {e}"))?;
        self.infer_image(&img, cfg)
    }

    fn infer_image(&self, img: &DynamicImage, cfg: &OcrConfig) -> Result<OcrResult, String> {
        let boxes = self.detect_text_boxes(img, cfg)?;

        let mut lines: Vec<(u32, u32, String, f32)> = Vec::new();
        for b in boxes {
            let crop = crop_box(img, b);
            let rec_input = preprocess_rec_image(&crop, cfg.rec_img_h, cfg.rec_img_w)?;
            let output = ort::run_session(&self.rec, &[rec_input])?;
            let logits = &output[0];
            let (text, confidence) = ctc_greedy_decode(logits, &self.alphabet);
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
            let output = ort::run_session(&self.rec, &[rec_input])?;
            let logits = &output[0];
            let (fallback_text, fallback_confidence) = ctc_greedy_decode(logits, &self.alphabet);
            text = fallback_text;
            confidence = fallback_confidence;

            if text.trim().is_empty()
                && let Some(rec_alt) = &self.rec_alt
            {
                let rec_input = preprocess_rec_image(img, cfg.rec_img_h, cfg.rec_img_w)?;
                let output = ort::run_session(rec_alt, &[rec_input])?;
                let logits = &output[0];
                let (alt_text, alt_confidence) = ctc_greedy_decode(logits, &self.alphabet_alt);
                text = alt_text;
                confidence = alt_confidence;
            }

            if text.trim().is_empty() {
                let mut line_texts = Vec::new();
                let mut confs = Vec::new();
                for line in fallback_line_crops(img) {
                    let rec_input = preprocess_rec_image(&line, cfg.rec_img_h, cfg.rec_img_w)?;
                    if let Ok(output) = ort::run_session(&self.rec, &[rec_input]) {
                        let logits = &output[0];
                        let (t, c) = ctc_greedy_decode(logits, &self.alphabet);
                        if !t.trim().is_empty() {
                            line_texts.push(t);
                            confs.push(c);
                        }
                    }
                }
                if !line_texts.is_empty() {
                    text = line_texts.join("\n");
                    confidence = confs.iter().sum::<f32>() / confs.len() as f32;
                }
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

type BoxRect = (u32, u32, u32, u32);

impl OrtOcrEngine {
    fn detect_text_boxes(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
    ) -> Result<Vec<BoxRect>, String> {
        let (det_input, sx, sy, w, h) = preprocess_det_image(img, cfg.det_img_side)?;
        let output = ort::run_session(&self.det, &[det_input])?;
        let map = &output[0];
        let boxes = extract_boxes_from_map(map, cfg.det_box_thresh, cfg.det_min_box_area, w, h);

        let min_w = (w as f32 * 0.015).ceil() as u32;
        let min_h = (h as f32 * 0.012).ceil() as u32;
        let mut scaled: Vec<BoxRect> = Vec::new();
        for b in boxes {
            let bw = (b.2 - b.0) as f32;
            let bh = (b.3 - b.1) as f32;
            let perimeter = (bw + bh) * 2.0;
            let area = bw * bh;
            let dist = if perimeter > 0.0 {
                area * 1.5 / perimeter
            } else {
                1.0f32
            };
            let h_expand = (dist * sx * 0.6) as u32;
            let v_expand = (dist * sy * 0.6) as u32;
            let x0 = (b.0 as f32 * sx).round() as i32 - h_expand as i32;
            let y0 = (b.1 as f32 * sy).round() as i32 - v_expand as i32;
            let x1 = ((b.2 as f32 * sx).round() as u32 + h_expand).min(w);
            let y1 = ((b.3 as f32 * sy).round() as u32 + v_expand).min(h);
            let x0 = x0.max(0) as u32;
            let y0 = y0.max(0) as u32;
            if x1 > x0 && y1 > y0 && x1 - x0 >= min_w && y1 - y0 >= min_h {
                scaled.push((x0, y0, x1, y1));
            }
        }

        scaled.sort_by_key(|(_, y0, _, _)| *y0);
        let mut merged: Vec<BoxRect> = Vec::new();
        for b in scaled {
            if let Some(last) = merged.last_mut() {
                let y_overlap = last.1 < b.3 && b.1 < last.3;
                let y_center_diff =
                    ((last.1 + last.3) as i32 - (b.1 + b.3) as i32).unsigned_abs() as u32;
                let line_h = (last.3 - last.1).max(b.3 - b.1).max(1);
                let x_gap = if b.0 > last.2 { b.0 - last.2 } else { 0 };
                if (y_overlap || y_center_diff <= line_h) && x_gap <= line_h * 3 {
                    last.0 = last.0.min(b.0);
                    last.1 = last.1.min(b.1);
                    last.2 = last.2.max(b.2);
                    last.3 = last.3.max(b.3);
                    continue;
                }
            }
            merged.push(b);
        }

        merged.retain(|(x0, y0, x1, y1)| x1 > x0 && y1 > y0);
        if merged.is_empty() {
            merged.push((0, 0, w, h));
        }
        Ok(merged)
    }
}

fn load_model_bytes(path: Option<&str>, embedded: &[u8]) -> Vec<u8> {
    if let Some(p) = path {
        std::fs::read(Path::new(p)).unwrap_or_else(|_| embedded.to_vec())
    } else {
        embedded.to_vec()
    }
}

fn load_ort_session(path: Option<&str>, embedded: &[u8]) -> Result<OrtSession, String> {
    let model_bytes = load_model_bytes(path, embedded);
    let session_ptr = ort::create_session_from_memory(&model_bytes)?;
    let allocator_ptr = ort::create_allocator()?;
    let memory_info_ptr = ort::create_memory_info()?;
    Ok(OrtSession {
        session_ptr,
        allocator_ptr,
        memory_info_ptr,
    })
}

fn load_dict(path: Option<&str>, embedded: &str) -> Vec<String> {
    let content = if let Some(p) = path {
        std::fs::read_to_string(p).unwrap_or_else(|_| embedded.to_string())
    } else {
        embedded.to_string()
    };
    content
        .lines()
        .map(|line| line.to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

fn preprocess_det_image(
    image: &DynamicImage,
    side: usize,
) -> Result<(Vec<f32>, f32, f32, u32, u32), String> {
    let rgb = to_rgb_on_white(image);
    let (src_w, src_h) = rgb.dimensions();
    let max_side = side as f32;
    let base = (src_w.max(src_h)) as f32;
    let ratio = if base > max_side { max_side / base } else { 1.0 };
    let resize_w = ((((src_w as f32) * ratio).round() as usize).max(32) / 32) * 32;
    let resize_h = ((((src_h as f32) * ratio).round() as usize).max(32) / 32) * 32;
    let resized = image::imageops::resize(
        &rgb,
        resize_w as u32,
        resize_h as u32,
        FilterType::Triangle,
    );

    let mut data = vec![0f32; 1 * 3 * side * side];
    for y in 0..resize_h {
        for x in 0..resize_w {
            let px = resized.get_pixel(x as u32, y as u32);
            let norm = [
                (px[2] as f32 / 255.0 - 0.485) / 0.229,
                (px[1] as f32 / 255.0 - 0.456) / 0.224,
                (px[0] as f32 / 255.0 - 0.406) / 0.225,
            ];
            for c in 0..3 {
                let idx = c * side * side + y * side + x;
                data[idx] = norm[c];
            }
        }
    }

    let sx = src_w as f32 / resize_w as f32;
    let sy = src_h as f32 / resize_h as f32;
    Ok((data, sx, sy, src_w, src_h))
}

fn extract_boxes_from_map(
    data: &[f32],
    thresh: f32,
    min_area: usize,
    map_w: u32,
    map_h: u32,
) -> Vec<BoxRect> {
    let h = map_h as usize;
    let w = map_w as usize;
    let mut mask = vec![false; h * w];
    for y in 0..h {
        for x in 0..w {
            mask[y * w + x] = data[y * w + x] >= thresh;
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

fn preprocess_rec_image(
    image: &DynamicImage,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<f32>, String> {
    let rgb = to_rgb_on_white(image);
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
                let v = (px[2 - c] as f32 / 255.0 - 0.5) / 0.5;
                let idx = c * target_h * target_w + y * target_w + x;
                data[idx] = v;
            }
        }
    }
    Ok(data)
}

fn to_rgb_on_white(image: &DynamicImage) -> image::RgbImage {
    let rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let r = (p[0] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            let g = (p[1] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            let b = (p[2] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    out
}

fn fallback_line_crops(image: &DynamicImage) -> Vec<DynamicImage> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut row_score = vec![0usize; h as usize];
    for y in 0..h {
        let mut c = 0usize;
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let lum = (p[0] as u16 + p[1] as u16 + p[2] as u16) / 3;
            if lum < 230 {
                c += 1;
            }
        }
        row_score[y as usize] = c;
    }
    let threshold = (w as usize / 80).max(12);
    let mut bands = Vec::new();
    let mut y = 0usize;
    while y < h as usize {
        if row_score[y] < threshold {
            y += 1;
            continue;
        }
        let start = y;
        let mut end = y;
        while end + 1 < h as usize && row_score[end + 1] >= threshold / 2 {
            end += 1;
        }
        y = end + 1;
        let height = end - start + 1;
        if !(10..=96).contains(&height) {
            continue;
        }
        let mut min_x = w;
        let mut max_x = 0u32;
        for yy in start as u32..=end as u32 {
            for xx in 0..w {
                let p = rgb.get_pixel(xx, yy);
                let lum = (p[0] as u16 + p[1] as u16 + p[2] as u16) / 3;
                if lum < 230 {
                    min_x = min_x.min(xx);
                    max_x = max_x.max(xx);
                }
            }
        }
        if max_x > min_x && (max_x - min_x) >= 24 {
            bands.push((min_x, start as u32, max_x + 1, end as u32 + 1));
        }
    }
    bands.sort_by_key(|(_, y0, _, _)| *y0);
    bands
        .into_iter()
        .take(48)
        .map(|(x0, y0, x1, y1)| image.crop_imm(x0, y0, (x1 - x0).max(1), (y1 - y0).max(1)))
        .collect()
}

fn ctc_greedy_decode(logits: &[f32], alphabet: &[String]) -> (String, f32) {
    let shape = g_outer_shape(logits);
    if shape.len() < 2 {
        return (String::new(), 0.0);
    }

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
                logits[c * steps + t]
            } else {
                logits[t * classes + c]
            };
            if v > best_val {
                best_val = v;
                best_id = c;
            }
        }
        if best_id != blank_id && best_id != prev {
            let idx = best_id.saturating_sub(1);
            if let Some(ch) = alphabet.get(idx) {
                if ch == "\u{3000}" {
                    continue;
                }
                text.push_str(ch);
                prob_sum += best_val;
                count += 1;
            }
        }
        prev = best_id;
    }
    let confidence = if count == 0 { 0.0 } else { prob_sum / count as f32 };
    (text, confidence)
}

fn g_outer_shape(data: &[f32]) -> Vec<usize> {
    let total = data.len();
    if total == 0 {
        return vec![1, 1, total];
    }
    let mut shape = ort::last_known_shape();
    let product: usize = shape.iter().skip(1).product();
    if product > 0 {
        shape[0] = total / product;
    } else {
        shape = vec![1, 1, total];
    }
    shape
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
    fn default_config_points_to_embedded_models() {
        let cfg = OcrConfig::default();
        assert!(cfg.det_model_path.is_none());
        assert!(cfg.rec_model_path.is_none());
    }
}
