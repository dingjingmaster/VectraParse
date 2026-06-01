use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, GrayImage, Luma};

mod ort;

const EMBED_DET_ONNX: &[u8] = include_bytes!("../../../data/det.onnx");
const EMBED_REC_ZH_ONNX: &[u8] = include_bytes!("../../../data/chinese/rec.onnx");
const EMBED_REC_EN_ONNX: &[u8] = include_bytes!("../../../data/english/rec.onnx");
const EMBED_DICT_ZH: &str = include_str!("../../../data/chinese/dict.txt");
const EMBED_DICT_EN: &str = include_str!("../../../data/english/dict.txt");
const MAX_UPSCALE_PIXELS: u64 = 2_500_000;
const MAX_REC_IMG_W: usize = 960;
const MIN_ACCEPT_REC_CONFIDENCE: f32 = 0.25;
const MIN_STRONG_REC_CONFIDENCE: f32 = 0.55;

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
            det_box_thresh: 0.20,
            det_min_box_area: 20,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
    pub warning: Option<String>,
    pub diagnostics: OcrDiagnostics,
}

#[derive(Debug, Clone, Default)]
pub struct OcrDiagnostics {
    pub det_box_count: usize,
    pub line_count: usize,
    pub fallback: Option<String>,
    pub empty_result: bool,
    pub source_has_alpha: bool,
    pub detect_used_whole_image_box: bool,
}

pub struct OrtOcrEngine {
    det: OrtSession,
    rec: OrtSession,
    rec_alt: Option<OrtSession>,
    alphabet: Vec<String>,
    alphabet_alt: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecVariant {
    Primary,
    Alt,
}

#[derive(Debug, Clone)]
struct RecCandidate {
    text: String,
    confidence: f32,
    variant: RecVariant,
}

#[derive(Debug, Clone, Default)]
struct RecognizedText {
    text: String,
    confidence: f32,
    line_count: usize,
}

#[derive(Debug, Clone, Default)]
struct DetectedText {
    det_box_count: usize,
    recognized: RecognizedText,
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
        let source_has_alpha = has_non_opaque_alpha(img);
        let detected = self
            .recognize_detected_text(img, cfg, true)
            .map_err(|e| format!("detect: {e}"))?;
        let det_box_count = detected.det_box_count;
        let detect_used_whole_image_box = det_box_count == 0;
        let mut text = detected.recognized.text;
        let mut confidence = detected.recognized.confidence;
        let mut line_count = detected.recognized.line_count;
        let mut fallback = None;

        if needs_quality_fallback(&text, confidence, det_box_count, line_count) {
            self.apply_quality_fallbacks(
                img,
                cfg,
                det_box_count,
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut fallback,
            )?;
        }

        let warning = if self.alphabet.is_empty() {
            Some("ocr-dictionary-missing".to_string())
        } else {
            None
        };
        let empty_result = text.trim().is_empty();

        if std::env::var("VECTRAPARSE_OCR_TRACE").ok().as_deref() == Some("1") {
            let (w, h) = img.dimensions();
            eprintln!(
                "[OCR_TRACE] dims={}x{} alpha={} det_boxes={} line_count={} whole_image_box={} fallback={} empty={}",
                w,
                h,
                source_has_alpha,
                det_box_count,
                line_count,
                detect_used_whole_image_box,
                fallback.as_deref().unwrap_or("-"),
                empty_result
            );
        }

        Ok(OcrResult {
            text,
            confidence,
            warning,
            diagnostics: OcrDiagnostics {
                det_box_count,
                line_count,
                fallback,
                empty_result,
                source_has_alpha,
                detect_used_whole_image_box,
            },
        })
    }

    fn recognize_detected_text(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        allow_crop_enhancement: bool,
    ) -> Result<DetectedText, String> {
        let boxes = self.detect_text_boxes(img, cfg)?;
        let mut lines: Vec<(u32, u32, String, f32)> = Vec::new();
        for b in boxes.iter() {
            let crop = crop_box(img, *b);
            let candidate = if allow_crop_enhancement {
                self.best_from_crop(&crop, cfg)
            } else {
                self.best_from_crop_direct(&crop, cfg)
            };
            if let Some(candidate) = candidate {
                lines.push((b.1, b.0, candidate.text, candidate.confidence));
            }
        }

        Ok(DetectedText {
            det_box_count: boxes.len(),
            recognized: recognized_from_positioned_lines(&mut lines),
        })
    }

    fn apply_quality_fallbacks(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        det_box_count: usize,
        text: &mut String,
        confidence: &mut f32,
        line_count: &mut usize,
        fallback: &mut Option<String>,
    ) -> Result<(), String> {
        match self.recognize_best(img, cfg) {
            Ok(candidate) if is_usable_recognition(&candidate) => {
                let label = recognition_fallback_label("whole-image", candidate.variant);
                let candidate = recognized_from_candidate(candidate);
                maybe_adopt_recognized(text, confidence, line_count, fallback, label, &candidate);
            }
            Err(e) if text.trim().is_empty() => return Err(format!("recognize: {e}")),
            _ => {}
        }

        if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
            return Ok(());
        }

        for (name, enhanced) in enhancement_variants(img) {
            if let Ok(candidate) = self.recognize_detected_text(&enhanced, cfg, false) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    fallback,
                    format!("det-enhanced:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            if let Ok(candidate) = self.recognize_best(&enhanced, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("enhanced:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate);
                maybe_adopt_recognized(text, confidence, line_count, fallback, label, &candidate);
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        for (name, upscaled) in upscale_variants(img) {
            if let Ok(candidate) = self.recognize_detected_text(&upscaled, cfg, false) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    fallback,
                    format!("det-upscaled:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            if let Ok(candidate) = self.recognize_best(&upscaled, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("upscaled:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate);
                maybe_adopt_recognized(text, confidence, line_count, fallback, label, &candidate);
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        for (name, rotated) in rotation_variants(img) {
            if let Ok(candidate) = self.recognize_detected_text(&rotated, cfg, false) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    fallback,
                    format!("det-rotated:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            if let Ok(candidate) = self.recognize_best(&rotated, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("rotated:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate);
                maybe_adopt_recognized(text, confidence, line_count, fallback, label, &candidate);
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        let candidate = self.recognize_line_crops(img, cfg);
        maybe_adopt_recognized(
            text,
            confidence,
            line_count,
            fallback,
            "line-crops".to_string(),
            &candidate,
        );

        Ok(())
    }

    fn recognize_line_crops(&self, img: &DynamicImage, cfg: &OcrConfig) -> RecognizedText {
        let mut lines = Vec::new();
        for (idx, line) in fallback_line_crops(img).into_iter().enumerate() {
            if let Ok(candidate) = self.recognize_best(&line, cfg)
                && is_usable_recognition(&candidate)
            {
                lines.push((idx as u32, 0, candidate.text, candidate.confidence));
            }
        }
        recognized_from_positioned_lines(&mut lines)
    }

    fn recognize_best(
        &self,
        image: &DynamicImage,
        cfg: &OcrConfig,
    ) -> Result<RecCandidate, String> {
        let primary =
            self.recognize_candidate(&self.rec, &self.alphabet, image, cfg, RecVariant::Primary)?;
        let alt = if let Some(rec_alt) = &self.rec_alt {
            Some(self.recognize_candidate(
                rec_alt,
                &self.alphabet_alt,
                image,
                cfg,
                RecVariant::Alt,
            )?)
        } else {
            None
        };
        Ok(select_recognition(primary, alt))
    }

    fn best_from_crop(&self, image: &DynamicImage, cfg: &OcrConfig) -> Option<RecCandidate> {
        let direct = self.recognize_best(image, cfg).ok();
        if let Some(candidate) = &direct {
            if is_usable_recognition(candidate) && candidate.confidence >= MIN_STRONG_REC_CONFIDENCE
            {
                return Some(candidate.clone());
            }
        }
        let mut best = direct.filter(is_usable_recognition);
        for (_name, enhanced) in enhancement_variants(image) {
            if let Ok(candidate) = self.recognize_best(&enhanced, cfg) {
                if is_usable_recognition(&candidate)
                    && best
                        .as_ref()
                        .map_or(true, |b| candidate.confidence > b.confidence)
                {
                    best = Some(candidate);
                }
            }
        }
        best
    }

    fn best_from_crop_direct(&self, image: &DynamicImage, cfg: &OcrConfig) -> Option<RecCandidate> {
        self.recognize_best(image, cfg)
            .ok()
            .filter(is_usable_recognition)
    }

    fn recognize_candidate(
        &self,
        session: &OrtSession,
        alphabet: &[String],
        image: &DynamicImage,
        cfg: &OcrConfig,
        variant: RecVariant,
    ) -> Result<RecCandidate, String> {
        let target_w = dynamic_rec_target_width(image, cfg.rec_img_h, cfg.rec_img_w);
        match self.recognize_candidate_at_width(session, alphabet, image, cfg, variant, target_w) {
            Ok(candidate) => Ok(candidate),
            Err(e) if target_w != cfg.rec_img_w => self
                .recognize_candidate_at_width(session, alphabet, image, cfg, variant, cfg.rec_img_w)
                .map_err(|fallback| format!("{e}; fixed-width fallback failed: {fallback}")),
            Err(e) => Err(e),
        }
    }

    fn recognize_candidate_at_width(
        &self,
        session: &OrtSession,
        alphabet: &[String],
        image: &DynamicImage,
        cfg: &OcrConfig,
        variant: RecVariant,
        target_w: usize,
    ) -> Result<RecCandidate, String> {
        let (rec_input, rec_shape) = preprocess_rec_image(image, cfg.rec_img_h, target_w)?;
        let (output, out_shapes) = ort::run_session(session, &[rec_input], &[rec_shape])?;
        let logits = &output[0];
        let (text, confidence) = ctc_greedy_decode(logits, &out_shapes[0], alphabet);
        Ok(RecCandidate {
            text,
            confidence,
            variant,
        })
    }
}

fn recognized_from_positioned_lines(lines: &mut Vec<(u32, u32, String, f32)>) -> RecognizedText {
    lines.sort_by_key(|(y, x, _, _)| (*y / 8, *x));
    let text = lines
        .iter()
        .map(|(_, _, t, _)| t.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let line_count = text_line_count(&text);
    let confidence = if lines.is_empty() {
        0.0
    } else {
        lines.iter().map(|(_, _, _, c)| *c).sum::<f32>() / lines.len() as f32
    };
    RecognizedText {
        text,
        confidence,
        line_count,
    }
}

fn recognized_from_candidate(candidate: RecCandidate) -> RecognizedText {
    let line_count = text_line_count(&candidate.text);
    RecognizedText {
        text: candidate.text,
        confidence: candidate.confidence,
        line_count,
    }
}

fn recognition_fallback_label(base: &str, variant: RecVariant) -> String {
    match variant {
        RecVariant::Primary => base.to_string(),
        RecVariant::Alt => format!("{base}:alt"),
    }
}

fn maybe_adopt_recognized(
    text: &mut String,
    confidence: &mut f32,
    line_count: &mut usize,
    fallback: &mut Option<String>,
    label: String,
    candidate: &RecognizedText,
) -> bool {
    if candidate.text.trim().is_empty() {
        return false;
    }

    if text.trim().is_empty() {
        *text = candidate.text.clone();
        *confidence = candidate.confidence;
        *line_count = candidate.line_count.max(text_line_count(text));
        *fallback = Some(label);
        return true;
    }

    let current_chars = recognized_char_count(text);
    let candidate_chars = recognized_char_count(&candidate.text);
    let current_is_weak = (*confidence > 0.0 && *confidence < 0.35)
        || current_chars < 4
        || (current_chars >= 4 && readable_ratio(text) < 0.55);
    if current_is_weak
        && candidate_chars > current_chars + 2
        && candidate.confidence + 0.10 >= *confidence
    {
        *text = candidate.text.clone();
        *confidence = candidate.confidence;
        *line_count = candidate.line_count.max(text_line_count(text));
        *fallback = Some(label);
        return true;
    }

    let merged = merge_unique_lines(text, &candidate.text);
    let merged_chars = recognized_char_count(&merged);
    if merged_chars > current_chars + 2 && candidate.confidence + 0.10 >= *confidence {
        *text = merged;
        *confidence = merge_confidence(
            *confidence,
            *line_count,
            candidate.confidence,
            candidate.line_count,
        );
        *line_count = text_line_count(text);
        *fallback = Some(format!("merged:{label}"));
        return true;
    }

    let candidate_is_longer = candidate_chars > current_chars + 2;
    let candidate_is_clearer =
        candidate.confidence > *confidence + 0.08 && candidate_chars + 2 >= current_chars;
    if (candidate_is_longer && candidate.confidence + 0.10 >= *confidence) || candidate_is_clearer {
        *text = candidate.text.clone();
        *confidence = candidate.confidence;
        *line_count = candidate.line_count.max(text_line_count(text));
        *fallback = Some(label);
        return true;
    }

    false
}

fn merge_confidence(
    current_confidence: f32,
    current_lines: usize,
    candidate_confidence: f32,
    candidate_lines: usize,
) -> f32 {
    let current_weight = current_lines.max(1) as f32;
    let candidate_weight = candidate_lines.max(1) as f32;
    (current_confidence * current_weight + candidate_confidence * candidate_weight)
        / (current_weight + candidate_weight)
}

fn merge_unique_lines(primary: &str, fallback: &str) -> String {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let mut seen = lines
        .iter()
        .map(|line| normalize_ocr_line(line))
        .collect::<Vec<_>>();

    for line in fallback
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let key = normalize_ocr_line(line);
        if key.is_empty() {
            continue;
        }
        let duplicate = seen.iter().any(|existing| {
            existing == &key
                || (existing.len().min(key.len()) >= 4
                    && (existing.contains(&key) || key.contains(existing)))
        });
        if duplicate {
            continue;
        }
        seen.push(key);
        lines.push(line.to_string());
    }

    lines.join("\n")
}

fn normalize_ocr_line(line: &str) -> String {
    line.chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn needs_quality_fallback(
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
) -> bool {
    let char_count = recognized_char_count(text);
    if char_count == 0 {
        return true;
    }
    if confidence > 0.0 && confidence < 0.35 {
        return true;
    }
    if det_box_count >= 4 && line_count * 2 <= det_box_count {
        return true;
    }
    if char_count < 4 && det_box_count >= 2 {
        return true;
    }
    if char_count >= 4 && readable_ratio(text) < 0.55 {
        return true;
    }
    false
}

fn recognized_char_count(text: &str) -> usize {
    text.chars().filter(|ch| !ch.is_whitespace()).count()
}

fn text_line_count(text: &str) -> usize {
    text.lines().filter(|line| !line.trim().is_empty()).count()
}

type BoxRect = (u32, u32, u32, u32);

impl OrtOcrEngine {
    fn detect_text_boxes(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
    ) -> Result<Vec<BoxRect>, String> {
        let (det_input, det_shape, sx, sy, src_w, src_h) =
            preprocess_det_image(img, cfg.det_img_side)?;
        let (det_output, _det_shapes) = ort::run_session(&self.det, &[det_input], &[det_shape])?;
        let map = &det_output[0];
        let shape = &_det_shapes[0];
        let map_h = shape.get(2).copied().unwrap_or(960) as u32;
        let map_w = shape.get(3).copied().unwrap_or(960) as u32;
        let boxes =
            extract_boxes_from_map(map, cfg.det_box_thresh, cfg.det_min_box_area, map_w, map_h);

        let min_w = (src_w as f32 * 0.015).ceil() as u32;
        let min_h = (src_h as f32 * 0.012).ceil() as u32;
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
            let x1 = ((b.2 as f32 * sx).round() as u32 + h_expand).min(src_w);
            let y1 = ((b.3 as f32 * sy).round() as u32 + v_expand).min(src_h);
            let x0 = x0.max(0) as u32;
            let y0 = y0.max(0) as u32;
            if x1 > x0 && y1 > y0 && x1 - x0 >= min_w && y1 - y0 >= min_h {
                scaled.push((x0, y0, x1, y1));
            }
        }

        let mut scaled = nms_boxes(scaled, 0.35);
        scaled.sort_by(|a, b| {
            let ya = a.1 as i32 / 8;
            let yb = b.1 as i32 / 8;
            ya.cmp(&yb).then_with(|| a.0.cmp(&b.0))
        });
        let mut merged: Vec<BoxRect> = Vec::new();
        for b in scaled {
            if let Some(last) = merged.last_mut() {
                let y_overlap = last.1 < b.3 && b.1 < last.3;
                let x_gap = if b.0 > last.2 { b.0 - last.2 } else { 0 };
                let line_h = (last.3 - last.1).min(b.3 - b.1);
                let new_w = last.2.max(b.2) - last.0.min(b.0);
                if y_overlap && x_gap <= (line_h * 3).max(20) && new_w <= 1200 {
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
) -> Result<(Vec<f32>, Vec<usize>, f32, f32, u32, u32), String> {
    let rgb = to_rgb_on_white(image);
    let (src_w, src_h) = rgb.dimensions();
    let max_side = side as f32;
    let base = (src_w.max(src_h)) as f32;
    let ratio = if base > max_side {
        max_side / base
    } else {
        1.0
    };
    let resize_w = ((((src_w as f32) * ratio).round() as usize).max(32) / 32) * 32;
    let resize_h = ((((src_h as f32) * ratio).round() as usize).max(32) / 32) * 32;
    let resized =
        image::imageops::resize(&rgb, resize_w as u32, resize_h as u32, FilterType::Triangle);

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
    let shape = vec![1, 3, side, side];
    Ok((data, shape, sx, sy, src_w, src_h))
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
    let dilated = dilate_mask(&mask, w, h);
    let mut boxes = collect_boxes_from_mask(data, &mask, &dilated, thresh, min_area, w, h);
    if boxes.is_empty() {
        boxes = collect_boxes_from_mask(data, &mask, &mask, thresh, min_area, w, h);
    }
    nms_boxes(boxes, 0.35)
}

fn dilate_mask(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let mut active = false;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h.saturating_sub(1));
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w.saturating_sub(1));
            'scan: for ny in y0..=y1 {
                for nx in x0..=x1 {
                    if mask[ny * w + nx] {
                        active = true;
                        break 'scan;
                    }
                }
            }
            out[y * w + x] = active;
        }
    }
    out
}

fn collect_boxes_from_mask(
    data: &[f32],
    raw_mask: &[bool],
    component_mask: &[bool],
    thresh: f32,
    min_area: usize,
    w: usize,
    h: usize,
) -> Vec<BoxRect> {
    let mut visited = vec![false; h * w];
    let mut boxes = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || !component_mask[idx] {
                continue;
            }
            let mut queue = vec![(x, y)];
            visited[idx] = true;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            while let Some((cx, cy)) = queue.pop() {
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
                    if visited[nidx] || !component_mask[nidx] {
                        continue;
                    }
                    visited[nidx] = true;
                    queue.push((nx, ny));
                }
            }

            let positive_area = count_mask_area(raw_mask, min_x, min_y, max_x, max_y, w);
            if positive_area < min_area {
                continue;
            }

            let score =
                average_component_score(data, component_mask, min_x, min_y, max_x, max_y, w);
            if score < (thresh * 0.3) {
                continue;
            }

            boxes.push(expand_box(min_x, min_y, max_x, max_y, w, h));
        }
    }
    boxes
}

fn count_mask_area(
    mask: &[bool],
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    w: usize,
) -> usize {
    let mut area = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if mask[y * w + x] {
                area += 1;
            }
        }
    }
    area
}

fn average_component_score(
    data: &[f32],
    component_mask: &[bool],
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    w: usize,
) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if component_mask[y * w + x] {
                sum += data[y * w + x].max(0.0);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn expand_box(
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    w: usize,
    h: usize,
) -> BoxRect {
    let rect_w = max_x.saturating_sub(min_x) + 1;
    let rect_h = max_y.saturating_sub(min_y) + 1;
    let pad = if rect_w.min(rect_h) > 4 {
        (rect_w.min(rect_h) / 6).clamp(1, 8)
    } else {
        0
    };
    let x0 = min_x.saturating_sub(pad);
    let y0 = min_y.saturating_sub(pad);
    let x1 = (max_x + 1 + pad).min(w);
    let y1 = (max_y + 1 + pad).min(h);
    (x0 as u32, y0 as u32, x1 as u32, y1 as u32)
}

fn nms_boxes(mut boxes: Vec<BoxRect>, iou_threshold: f32) -> Vec<BoxRect> {
    boxes.retain(|b| b.2 > b.0 && b.3 > b.1);
    boxes.sort_by(|a, b| box_area(*b).cmp(&box_area(*a)));

    let mut kept: Vec<BoxRect> = Vec::new();
    for b in boxes {
        if kept
            .iter()
            .all(|kept_box| box_iou(b, *kept_box) <= iou_threshold)
        {
            kept.push(b);
        }
    }
    kept.sort_by_key(|b| (b.1, b.0));
    kept
}

fn box_area(b: BoxRect) -> u64 {
    (b.2.saturating_sub(b.0) as u64).saturating_mul(b.3.saturating_sub(b.1) as u64)
}

fn box_iou(a: BoxRect, b: BoxRect) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }
    let inter = box_area((x0, y0, x1, y1)) as f32;
    let union = (box_area(a) + box_area(b)) as f32 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
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
) -> Result<(Vec<f32>, Vec<usize>), String> {
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
    let shape = vec![1, 3, target_h, target_w];
    Ok((data, shape))
}

fn dynamic_rec_target_width(image: &DynamicImage, target_h: usize, base_w: usize) -> usize {
    let gray = to_luma_on_white(image);
    let (src_w, src_h) = gray.dimensions();
    if src_w == 0 || src_h == 0 {
        return base_w.max(1);
    }
    let ratio = src_w as f32 / src_h as f32;
    let raw = (ratio * target_h as f32).ceil() as usize;
    raw.max(base_w).clamp(1, MAX_REC_IMG_W.max(base_w))
}

fn to_rgb_on_white(image: &DynamicImage) -> image::RgbImage {
    to_rgb_on_background(image, 255)
}

fn to_rgb_on_background(image: &DynamicImage, background: u8) -> image::RgbImage {
    let rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = image::RgbImage::new(w, h);
    let bg = background as f32;
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let r = (p[0] as f32 * a + bg * (1.0 - a)).round() as u8;
            let g = (p[1] as f32 * a + bg * (1.0 - a)).round() as u8;
            let b = (p[2] as f32 * a + bg * (1.0 - a)).round() as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    out
}

fn to_luma_on_white(image: &DynamicImage) -> GrayImage {
    to_luma_on_background(image, 255)
}

fn to_luma_on_background(image: &DynamicImage, background: u8) -> GrayImage {
    DynamicImage::ImageRgb8(to_rgb_on_background(image, background)).to_luma8()
}

fn has_non_opaque_alpha(image: &DynamicImage) -> bool {
    let rgba = image.to_rgba8();
    rgba.pixels().any(|p| p[3] < 255)
}

fn to_hsl_lightness(image: &DynamicImage) -> GrayImage {
    to_hsl_lightness_on_background(image, 255)
}

fn to_hsl_lightness_on_background(image: &DynamicImage, background: u8) -> GrayImage {
    let rgb = to_rgb_on_background(image, background);
    let (w, h) = rgb.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let max_c = p[0].max(p[1]).max(p[2]);
            let min_c = p[0].min(p[1]).min(p[2]);
            out.put_pixel(x, y, Luma([((max_c as u16 + min_c as u16) / 2) as u8]));
        }
    }
    out
}

fn to_max_channel_gray(image: &DynamicImage) -> GrayImage {
    to_max_channel_gray_on_background(image, 255)
}

fn to_max_channel_gray_on_background(image: &DynamicImage, background: u8) -> GrayImage {
    let rgb = to_rgb_on_background(image, background);
    let (w, h) = rgb.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            out.put_pixel(x, y, Luma([p[0].max(p[1]).max(p[2])]));
        }
    }
    out
}

fn enhancement_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let mut out = Vec::new();
    let mut bases = vec![
        ("".to_string(), to_luma_on_white(image)),
        ("hsl-".to_string(), to_hsl_lightness(image)),
        ("max-".to_string(), to_max_channel_gray(image)),
    ];
    if has_non_opaque_alpha(image) {
        bases.extend([
            ("alpha-black-".to_string(), to_luma_on_background(image, 0)),
            (
                "alpha-black-hsl-".to_string(),
                to_hsl_lightness_on_background(image, 0),
            ),
            (
                "alpha-black-max-".to_string(),
                to_max_channel_gray_on_background(image, 0),
            ),
        ]);
    }
    for (prefix, base) in bases {
        push_enhancement_variants(&mut out, &prefix, &base);
    }
    out
}

fn push_enhancement_variants(
    out: &mut Vec<(String, DynamicImage)>,
    prefix: &str,
    base: &GrayImage,
) {
    let stretched = contrast_stretch_luma(base);
    let binary = adaptive_binary_luma(&stretched, false);
    let binary_invert = adaptive_binary_luma(&stretched, true);
    let local_binary = local_binary_luma(&stretched, false);
    let local_binary_invert = local_binary_luma(&stretched, true);
    out.push((
        format!("{prefix}contrast"),
        DynamicImage::ImageLuma8(stretched),
    ));
    out.push((format!("{prefix}binary"), DynamicImage::ImageLuma8(binary)));
    out.push((
        format!("{prefix}binary-invert"),
        DynamicImage::ImageLuma8(binary_invert),
    ));
    out.push((
        format!("{prefix}local-binary"),
        DynamicImage::ImageLuma8(local_binary),
    ));
    out.push((
        format!("{prefix}local-binary-invert"),
        DynamicImage::ImageLuma8(local_binary_invert),
    ));
}

fn upscale_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let (w, h) = image.dimensions();
    let base_pixels = (w as u64).saturating_mul(h as u64);
    let is_small = w < 640 || h < 160 || base_pixels < 160_000;
    if !is_small || w == 0 || h == 0 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (name, scale) in [("1.5x", 1.5f32), ("2x", 2.0f32)] {
        let target_w = ((w as f32) * scale).round() as u32;
        let target_h = ((h as f32) * scale).round() as u32;
        let pixels = (target_w as u64).saturating_mul(target_h as u64);
        if pixels > MAX_UPSCALE_PIXELS {
            continue;
        }
        let resized = image::imageops::resize(image, target_w, target_h, FilterType::CatmullRom);
        out.push((name.to_string(), DynamicImage::ImageRgba8(resized)));
    }
    out
}

fn contrast_stretch_luma(gray: &GrayImage) -> GrayImage {
    let mut min_v = u8::MAX;
    let mut max_v = u8::MIN;
    for pixel in gray.pixels() {
        min_v = min_v.min(pixel[0]);
        max_v = max_v.max(pixel[0]);
    }
    if max_v <= min_v.saturating_add(8) {
        return gray.clone();
    }
    let range = (max_v - min_v) as u16;
    let mut out = GrayImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let raw = pixel[0].saturating_sub(min_v) as u16;
        let stretched = ((raw * 255) / range) as u8;
        out.put_pixel(x, y, Luma([stretched]));
    }
    out
}

fn rotation_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    vec![
        ("90".to_string(), image.rotate90()),
        ("180".to_string(), image.rotate180()),
        ("270".to_string(), image.rotate270()),
    ]
}

fn adaptive_binary_luma(gray: &GrayImage, invert: bool) -> GrayImage {
    let threshold = otsu_threshold_luma(gray).clamp(32, 223);
    binary_with_threshold(gray, threshold, invert)
}

fn binary_with_threshold(gray: &GrayImage, threshold: u8, invert: bool) -> GrayImage {
    let mut out = GrayImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let is_fg = if invert {
            pixel[0] >= threshold
        } else {
            pixel[0] <= threshold
        };
        let value = if is_fg { 0 } else { 255 };
        out.put_pixel(x, y, Luma([value]));
    }
    out
}

fn otsu_threshold_luma(gray: &GrayImage) -> u8 {
    let total = gray.width() as u64 * gray.height() as u64;
    if total == 0 {
        return 127;
    }
    let mut hist = [0u64; 256];
    for pixel in gray.pixels() {
        hist[pixel[0] as usize] += 1;
    }

    let mut sum_total = 0u64;
    for (value, count) in hist.iter().enumerate() {
        sum_total += value as u64 * count;
    }

    let mut weight_bg = 0u64;
    let mut sum_bg = 0u64;
    let mut best_threshold = 127u8;
    let mut best_variance = -1.0f64;
    for (threshold, count) in hist.iter().enumerate() {
        weight_bg += count;
        if weight_bg == 0 {
            continue;
        }
        let weight_fg = total.saturating_sub(weight_bg);
        if weight_fg == 0 {
            break;
        }
        sum_bg += threshold as u64 * count;
        let mean_bg = sum_bg as f64 / weight_bg as f64;
        let mean_fg = (sum_total - sum_bg) as f64 / weight_fg as f64;
        let diff = mean_bg - mean_fg;
        let variance = weight_bg as f64 * weight_fg as f64 * diff * diff;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = threshold as u8;
        }
    }
    best_threshold
}

fn local_binary_luma(gray: &GrayImage, invert: bool) -> GrayImage {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return GrayImage::new(w, h);
    }

    let iw = (w + 1) as usize;
    let ih = (h + 1) as usize;
    let mut integral = vec![0u64; iw * ih];
    for y in 0..h as usize {
        let mut row_sum = 0u64;
        for x in 0..w as usize {
            row_sum += gray.get_pixel(x as u32, y as u32)[0] as u64;
            integral[(y + 1) * iw + x + 1] = integral[y * iw + x + 1] + row_sum;
        }
    }

    let radius = 12usize;
    let mut out = GrayImage::new(w, h);
    for y in 0..h as usize {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(h as usize);
        for x in 0..w as usize {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(w as usize);
            let area = (x1 - x0) * (y1 - y0);
            let sum = integral[y1 * iw + x1] + integral[y0 * iw + x0]
                - integral[y0 * iw + x1]
                - integral[y1 * iw + x0];
            let mean = (sum / area.max(1) as u64) as i16;
            let threshold = (mean + if invert { 8 } else { -8 }).clamp(24, 231) as u8;
            let pixel = gray.get_pixel(x as u32, y as u32)[0];
            let is_fg = if invert {
                pixel >= threshold
            } else {
                pixel <= threshold
            };
            out.put_pixel(x as u32, y as u32, Luma([if is_fg { 0 } else { 255 }]));
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

fn ctc_greedy_decode(logits: &[f32], out_shape: &[usize], alphabet: &[String]) -> (String, f32) {
    let shape = g_outer_shape(logits, out_shape);
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
    let confidence = if count == 0 {
        0.0
    } else {
        prob_sum / count as f32
    };
    (text, confidence)
}

fn select_recognition(primary: RecCandidate, alt: Option<RecCandidate>) -> RecCandidate {
    let Some(alt) = alt else {
        return primary;
    };
    let primary_empty = primary.text.trim().is_empty();
    let alt_empty = alt.text.trim().is_empty();
    if primary_empty && !alt_empty {
        return alt;
    }
    if alt_empty {
        return primary;
    }

    let primary_ascii = ascii_ratio(&primary.text);
    let alt_ascii = ascii_ratio(&alt.text);
    if alt_ascii >= 0.75 && primary_ascii <= 0.5 && alt.confidence + 0.02 >= primary.confidence {
        return alt;
    }
    if alt_ascii >= 0.75 && primary_ascii >= 0.75 && alt.confidence > primary.confidence {
        return alt;
    }
    if alt.confidence > primary.confidence + 0.08 {
        return alt;
    }
    primary
}

fn is_usable_recognition(candidate: &RecCandidate) -> bool {
    let text = candidate.text.trim();
    if text.is_empty() || candidate.confidence < MIN_ACCEPT_REC_CONFIDENCE {
        return false;
    }

    let readable = readable_ratio(text);
    let repeat = dominant_char_ratio(text);
    let punct = punctuation_ratio(text);
    let char_count = text.chars().filter(|c| !c.is_whitespace()).count();

    if readable < 0.45 {
        return false;
    }
    if char_count >= 4 && repeat >= 0.78 {
        return false;
    }
    if char_count >= 4 && punct >= 0.75 {
        return false;
    }
    true
}

fn ascii_ratio(text: &str) -> f32 {
    let mut total = 0usize;
    let mut ascii = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if ch.is_ascii_alphanumeric() || ch.is_ascii_punctuation() {
            ascii += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        ascii as f32 / total as f32
    }
}

fn readable_ratio(text: &str) -> f32 {
    let mut total = 0usize;
    let mut readable = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if is_readable_ocr_char(ch) {
            readable += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        readable as f32 / total as f32
    }
}

fn is_readable_ocr_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || ch.is_ascii_punctuation()
        || ('\u{4E00}'..='\u{9FFF}').contains(&ch)
        || ('\u{3000}'..='\u{303F}').contains(&ch)
        || ('\u{FF00}'..='\u{FFEF}').contains(&ch)
}

fn dominant_char_ratio(text: &str) -> f32 {
    let mut counts: Vec<(char, usize)> = Vec::new();
    let mut total = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if let Some((_, count)) = counts.iter_mut().find(|(seen, _)| *seen == ch) {
            *count += 1;
        } else {
            counts.push((ch, 1));
        }
    }
    if total == 0 {
        0.0
    } else {
        counts.iter().map(|(_, count)| *count).max().unwrap_or(0) as f32 / total as f32
    }
}

fn punctuation_ratio(text: &str) -> f32 {
    let mut total = 0usize;
    let mut punct = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if ch.is_ascii_punctuation() || ('\u{3000}'..='\u{303F}').contains(&ch) {
            punct += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        punct as f32 / total as f32
    }
}

fn g_outer_shape(data: &[f32], output_shape: &[usize]) -> Vec<usize> {
    let total = data.len();
    let mut shape = output_shape.to_vec();
    let product: usize = shape.iter().skip(1).product();
    if product > 0 && total > 0 {
        shape[0] = total / product;
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

    #[test]
    fn enhancement_variants_include_expected_modes() {
        let mut gray = GrayImage::new(4, 2);
        gray.put_pixel(0, 0, Luma([96]));
        gray.put_pixel(1, 0, Luma([112]));
        gray.put_pixel(2, 0, Luma([144]));
        gray.put_pixel(3, 0, Luma([160]));
        gray.put_pixel(0, 1, Luma([100]));
        gray.put_pixel(1, 1, Luma([118]));
        gray.put_pixel(2, 1, Luma([146]));
        gray.put_pixel(3, 1, Luma([164]));
        let variants = enhancement_variants(&DynamicImage::ImageLuma8(gray));
        let names = variants
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                "contrast",
                "binary",
                "binary-invert",
                "local-binary",
                "local-binary-invert",
                "hsl-contrast",
                "hsl-binary",
                "hsl-binary-invert",
                "hsl-local-binary",
                "hsl-local-binary-invert",
                "max-contrast",
                "max-binary",
                "max-binary-invert",
                "max-local-binary",
                "max-local-binary-invert",
            ]
        );
    }

    #[test]
    fn enhancement_variants_include_black_background_for_alpha() {
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, image::Rgba([255, 255, 255, 255]));
        rgba.put_pixel(1, 0, image::Rgba([255, 255, 255, 0]));
        let variants = enhancement_variants(&DynamicImage::ImageRgba8(rgba));
        let names = variants
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), 30);
        assert!(names.contains(&"alpha-black-contrast"));
        assert!(names.contains(&"alpha-black-hsl-local-binary"));
        assert!(names.contains(&"alpha-black-max-binary-invert"));
    }

    #[test]
    fn contrast_stretch_expands_low_contrast_range() {
        let mut gray = GrayImage::new(2, 2);
        gray.put_pixel(0, 0, Luma([110]));
        gray.put_pixel(1, 0, Luma([115]));
        gray.put_pixel(0, 1, Luma([120]));
        gray.put_pixel(1, 1, Luma([125]));
        let stretched = contrast_stretch_luma(&gray);
        let values = stretched.pixels().map(|p| p[0]).collect::<Vec<_>>();
        assert_eq!(values.iter().min().copied(), Some(0));
        assert_eq!(values.iter().max().copied(), Some(255));
    }

    #[test]
    fn adaptive_binary_can_flip_for_light_foreground() {
        let mut gray = GrayImage::new(3, 1);
        gray.put_pixel(0, 0, Luma([20]));
        gray.put_pixel(1, 0, Luma([240]));
        gray.put_pixel(2, 0, Luma([25]));
        let normal = adaptive_binary_luma(&gray, false);
        let inverted = adaptive_binary_luma(&gray, true);
        assert_eq!(normal.get_pixel(1, 0)[0], 255);
        assert_eq!(inverted.get_pixel(1, 0)[0], 0);
    }

    #[test]
    fn otsu_threshold_splits_bimodal_luma() {
        let mut gray = GrayImage::new(8, 1);
        for x in 0..4 {
            gray.put_pixel(x, 0, Luma([24]));
        }
        for x in 4..8 {
            gray.put_pixel(x, 0, Luma([220]));
        }
        let threshold = otsu_threshold_luma(&gray);
        assert!((24..=220).contains(&threshold));
    }

    #[test]
    fn local_binary_preserves_shadowed_dark_text() {
        let mut gray = GrayImage::from_pixel(32, 8, Luma([180]));
        for x in 4..12 {
            gray.put_pixel(x, 3, Luma([120]));
        }
        for x in 20..28 {
            gray.put_pixel(x, 3, Luma([80]));
        }
        let binary = local_binary_luma(&gray, false);
        assert_eq!(binary.get_pixel(6, 3)[0], 0);
        assert_eq!(binary.get_pixel(24, 3)[0], 0);
    }

    #[test]
    fn upscale_variants_generated_for_small_images_only() {
        let small = DynamicImage::ImageLuma8(GrayImage::from_pixel(120, 40, Luma([128])));
        let large = DynamicImage::ImageLuma8(GrayImage::from_pixel(1200, 800, Luma([128])));
        let small_names = upscale_variants(&small)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(small_names, vec!["1.5x".to_string(), "2x".to_string()]);
        assert!(upscale_variants(&large).is_empty());
    }

    #[test]
    fn upscale_variants_respect_max_pixel_budget() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(5000, 150, Luma([128])));
        let names = upscale_variants(&img)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["1.5x".to_string()]);
    }

    #[test]
    fn extract_boxes_from_map_merges_nearby_fragments_after_dilation() {
        let map_w = 8;
        let map_h = 4;
        let mut data = vec![0.0f32; (map_w * map_h) as usize];
        for x in [1usize, 2, 4, 5] {
            data[map_w as usize + x] = 0.9;
        }
        let boxes = extract_boxes_from_map(&data, 0.5, 1, map_w, map_h);
        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn extract_boxes_from_map_keeps_raw_fallback_for_tiny_high_score_blob() {
        let map_w = 6;
        let map_h = 6;
        let mut data = vec![0.0f32; (map_w * map_h) as usize];
        data[2 * map_w as usize + 2] = 0.95;
        let boxes = extract_boxes_from_map(&data, 0.5, 1, map_w, map_h);
        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn extract_boxes_from_map_adds_small_crop_margin() {
        let map_w = 6;
        let map_h = 6;
        let mut data = vec![0.0f32; (map_w * map_h) as usize];
        for y in 2usize..=3 {
            for x in 2usize..=3 {
                data[y * map_w as usize + x] = 0.9;
            }
        }
        let boxes = extract_boxes_from_map(&data, 0.5, 1, map_w, map_h);
        assert_eq!(boxes, vec![(1, 1, 5, 5)]);
    }

    #[test]
    fn nms_boxes_removes_overlapping_boxes() {
        let boxes = vec![(0, 0, 12, 12), (1, 1, 11, 11), (30, 0, 40, 10)];
        let kept = nms_boxes(boxes, 0.35);
        assert_eq!(kept, vec![(0, 0, 12, 12), (30, 0, 40, 10)]);
    }

    #[test]
    fn expand_box_uses_capped_unclip_margin() {
        assert_eq!(expand_box(20, 20, 79, 79, 100, 100), (12, 12, 88, 88));
    }

    #[test]
    fn dynamic_rec_target_width_grows_for_long_lines() {
        let long = DynamicImage::ImageLuma8(GrayImage::from_pixel(1200, 48, Luma([128])));
        assert_eq!(dynamic_rec_target_width(&long, 48, 320), 960);
        let narrow = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 48, Luma([128])));
        assert_eq!(dynamic_rec_target_width(&narrow, 48, 320), 320);
    }

    #[test]
    fn select_recognition_can_choose_alt_for_ascii_line() {
        let primary = RecCandidate {
            text: "川川川".to_string(),
            confidence: 0.61,
            variant: RecVariant::Primary,
        };
        let alt = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.60,
            variant: RecVariant::Alt,
        };
        let chosen = select_recognition(primary, Some(alt));
        assert_eq!(chosen.variant, RecVariant::Alt);
        assert_eq!(chosen.text, "Invoice 42");
    }

    #[test]
    fn select_recognition_prefers_primary_when_alt_is_not_clear_win() {
        let primary = RecCandidate {
            text: "测试文本".to_string(),
            confidence: 0.68,
            variant: RecVariant::Primary,
        };
        let alt = RecCandidate {
            text: "Test Text".to_string(),
            confidence: 0.60,
            variant: RecVariant::Alt,
        };
        let chosen = select_recognition(primary.clone(), Some(alt));
        assert_eq!(chosen.variant, primary.variant);
        assert_eq!(chosen.text, primary.text);
    }

    #[test]
    fn usable_recognition_rejects_low_quality_text() {
        let repeated = RecCandidate {
            text: "||||||".to_string(),
            confidence: 0.91,
            variant: RecVariant::Primary,
        };
        let low_confidence = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.12,
            variant: RecVariant::Alt,
        };
        let valid = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.42,
            variant: RecVariant::Alt,
        };
        assert!(!is_usable_recognition(&repeated));
        assert!(!is_usable_recognition(&low_confidence));
        assert!(is_usable_recognition(&valid));
    }

    #[test]
    fn quality_fallback_detects_partial_success() {
        assert!(needs_quality_fallback("AB", 0.62, 3, 1));
        assert!(needs_quality_fallback("Invoice", 0.22, 1, 1));
        assert!(needs_quality_fallback("One line", 0.62, 6, 2));
        assert!(!needs_quality_fallback("Invoice 42", 0.62, 2, 2));
    }

    #[test]
    fn maybe_adopt_recognized_merges_unique_lines() {
        let mut text = "Header".to_string();
        let mut confidence = 0.44;
        let mut line_count = 1;
        let mut fallback = None;
        let candidate = RecognizedText {
            text: "Header\nTotal 42".to_string(),
            confidence: 0.54,
            line_count: 2,
        };
        assert!(maybe_adopt_recognized(
            &mut text,
            &mut confidence,
            &mut line_count,
            &mut fallback,
            "det-enhanced:contrast".to_string(),
            &candidate,
        ));
        assert_eq!(text, "Header\nTotal 42");
        assert_eq!(line_count, 2);
        assert_eq!(fallback.as_deref(), Some("merged:det-enhanced:contrast"));
    }

    #[test]
    fn rotation_variants_include_expected_angles() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(12, 8, Luma([128])));
        let variants = rotation_variants(&img);
        let names = variants
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["90", "180", "270"]);
        assert_eq!(variants[0].1.dimensions(), (8, 12));
        assert_eq!(variants[1].1.dimensions(), (12, 8));
        assert_eq!(variants[2].1.dimensions(), (8, 12));
    }

    #[test]
    fn detects_non_opaque_alpha_pixels() {
        let mut rgba = image::RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 0, 255]));
        rgba.put_pixel(1, 0, image::Rgba([0, 0, 0, 32]));
        assert!(has_non_opaque_alpha(&DynamicImage::ImageRgba8(rgba)));
    }

    #[test]
    fn rec_luma_blends_transparent_pixels_onto_white() {
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([0, 0, 0, 0]));
        let gray = to_luma_on_white(&DynamicImage::ImageRgba8(rgba));
        assert_eq!(gray.get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn luma_blends_transparent_pixels_onto_black() {
        let mut rgba = image::RgbaImage::new(1, 1);
        rgba.put_pixel(0, 0, image::Rgba([255, 255, 255, 0]));
        let gray = to_luma_on_background(&DynamicImage::ImageRgba8(rgba), 0);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
    }
}
