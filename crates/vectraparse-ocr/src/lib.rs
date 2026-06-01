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
const MAX_COLOR_REGION_PIXELS: u64 = 1_500_000;
const MAX_COLOR_REGION_CANDIDATES: usize = 48;
const MAX_EAGER_COLOR_REGION_RECOGNITIONS: usize = 8;
const MAX_REC_IMG_W: usize = 640;
const MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS: usize = 4;
const MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS: usize = 6;
const MAX_HORIZONTAL_SEGMENTS_PER_LINE: usize = 4;
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
    pub regions: Vec<OcrTextRegion>,
    pub trace: OcrTrace,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTextRegion {
    pub bbox: [u32; 4],
    pub text: String,
    pub confidence: f32,
    pub source: String,
    pub lines: Vec<OcrTextLine>,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTextLine {
    pub bbox: [u32; 4],
    pub text: String,
    pub confidence: f32,
    pub source: String,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTrace {
    pub selected_source: Option<String>,
    pub det_pass_count: usize,
    pub fallback_attempt_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct OcrDiagnostics {
    pub det_box_count: usize,
    pub line_count: usize,
    pub region_count: usize,
    pub layout_applied: bool,
    pub color_region_count: usize,
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
    region_count: usize,
    layout_applied: bool,
    regions: Vec<OcrTextRegion>,
}

#[derive(Debug, Clone, Default)]
struct DetectedText {
    det_box_count: usize,
    recognized: RecognizedText,
}

#[derive(Debug, Clone)]
struct TextLine {
    bbox: BoxRect,
    text: String,
    confidence: f32,
    source: String,
}

#[derive(Debug, Clone, Default)]
struct LayoutRegion {
    bbox: BoxRect,
    lines: Vec<TextLine>,
}

#[derive(Debug, Clone, Copy)]
enum BboxTransform {
    Identity,
    Scale {
        sx: f32,
        sy: f32,
        max_w: u32,
        max_h: u32,
    },
    Rotate90 {
        src_w: u32,
        src_h: u32,
    },
    Rotate180 {
        src_w: u32,
        src_h: u32,
    },
    Rotate270 {
        src_w: u32,
        src_h: u32,
    },
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
        let trace_enabled = ocr_trace_enabled();
        let source_has_alpha = has_non_opaque_alpha(img);
        if trace_enabled {
            let (w, h) = img.dimensions();
            eprintln!("[OCR_TRACE] start dims={w}x{h} alpha={source_has_alpha}");
        }
        let detected = self
            .recognize_detected_text(img, cfg, true, "det", BboxTransform::Identity)
            .map_err(|e| format!("detect: {e}"))?;
        let det_box_count = detected.det_box_count;
        let detect_used_whole_image_box = det_box_count == 0;
        let mut text = detected.recognized.text;
        let mut confidence = detected.recognized.confidence;
        let mut line_count = detected.recognized.line_count;
        let mut region_count = detected.recognized.region_count;
        let mut layout_applied = detected.recognized.layout_applied;
        let mut regions = detected.recognized.regions;
        let mut fallback = None;
        let mut trace = OcrTrace {
            selected_source: if text.trim().is_empty() {
                None
            } else {
                Some("det".to_string())
            },
            det_pass_count: 1,
            fallback_attempt_count: 0,
        };

        let (candidate_count, candidate) = self.recognize_color_regions_limited(
            img,
            cfg,
            MAX_EAGER_COLOR_REGION_RECOGNITIONS,
            "color-region:eager",
        );
        let mut color_region_count = candidate_count;
        let candidate = if text.trim().is_empty() {
            candidate
        } else {
            filter_non_overlapping_recognized(&candidate, &regions)
        };
        maybe_adopt_recognized(
            &mut text,
            &mut confidence,
            &mut line_count,
            &mut region_count,
            &mut layout_applied,
            &mut regions,
            &mut fallback,
            "color-regions:eager".to_string(),
            &candidate,
        );

        if needs_quality_fallback(&text, confidence, det_box_count, line_count) {
            self.apply_quality_fallbacks(
                img,
                cfg,
                det_box_count,
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut color_region_count,
                &mut trace,
                &mut fallback,
            )?;
        }

        let warning = if self.alphabet.is_empty() {
            Some("ocr-dictionary-missing".to_string())
        } else {
            None
        };
        let empty_result = text.trim().is_empty();
        trace.selected_source = fallback.clone().or_else(|| {
            if empty_result {
                None
            } else {
                Some("det".to_string())
            }
        });

        if trace_enabled {
            let (w, h) = img.dimensions();
            eprintln!(
                "[OCR_TRACE] dims={}x{} alpha={} det_boxes={} line_count={} regions={} layout={} color_regions={} det_passes={} fallback_attempts={} source={} whole_image_box={} fallback={} empty={}",
                w,
                h,
                source_has_alpha,
                det_box_count,
                line_count,
                region_count,
                layout_applied,
                color_region_count,
                trace.det_pass_count,
                trace.fallback_attempt_count,
                trace.selected_source.as_deref().unwrap_or("-"),
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
                region_count,
                layout_applied,
                color_region_count,
                fallback,
                empty_result,
                source_has_alpha,
                detect_used_whole_image_box,
            },
            regions,
            trace,
        })
    }

    fn recognize_detected_text(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        allow_crop_enhancement: bool,
        source: &str,
        transform: BboxTransform,
    ) -> Result<DetectedText, String> {
        let boxes = self.detect_text_boxes(img, cfg)?;
        let trace_enabled = ocr_trace_enabled();
        if trace_enabled {
            eprintln!(
                "[OCR_TRACE] det-pass source={} boxes={} crop_enhance_budget={} split_budget={}",
                source,
                boxes.len(),
                if allow_crop_enhancement {
                    MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS
                } else {
                    0
                },
                if source == "det" {
                    MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS
                } else {
                    0
                }
            );
        }
        let mut lines = Vec::new();
        let mut crop_enhancement_budget = if allow_crop_enhancement {
            MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS
        } else {
            0
        };
        let mut split_line_rec_budget = if source == "det" {
            MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS
        } else {
            0
        };
        for (idx, b) in boxes.iter().enumerate() {
            if trace_enabled {
                eprintln!(
                    "[OCR_TRACE] det-pass-box source={} index={} bbox={}x{}@{},{}",
                    source,
                    idx + 1,
                    box_width(*b),
                    box_height(*b),
                    b.0,
                    b.1
                );
            }
            self.push_recognized_box_lines(
                img,
                cfg,
                *b,
                allow_crop_enhancement,
                source,
                transform,
                &mut crop_enhancement_budget,
                &mut split_line_rec_budget,
                &mut lines,
            );
            if trace_enabled && (idx + 1) % 16 == 0 {
                eprintln!(
                    "[OCR_TRACE] det-pass-progress source={} processed={}/{} lines={}",
                    source,
                    idx + 1,
                    boxes.len(),
                    lines.len()
                );
            }
        }
        if trace_enabled {
            eprintln!(
                "[OCR_TRACE] det-pass-done source={} boxes={} lines={}",
                source,
                boxes.len(),
                lines.len()
            );
        }

        Ok(DetectedText {
            det_box_count: boxes.len(),
            recognized: recognized_from_text_lines(&mut lines),
        })
    }

    fn push_recognized_box_lines(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        b: BoxRect,
        allow_crop_enhancement: bool,
        source: &str,
        transform: BboxTransform,
        crop_enhancement_budget: &mut usize,
        split_line_rec_budget: &mut usize,
        lines: &mut Vec<TextLine>,
    ) {
        let mut split_boxes = split_text_box_into_color_region_boxes(img, b);
        if split_boxes.len() < 2 || split_boxes.len() > *split_line_rec_budget {
            split_boxes = split_text_box_into_line_boxes(img, b);
        }
        if split_boxes.len() >= 2 && split_boxes.len() <= *split_line_rec_budget {
            *split_line_rec_budget -= split_boxes.len();
            let split_lines =
                self.recognize_split_line_boxes(img, cfg, &split_boxes, source, transform);
            if should_use_split_lines(None, &split_lines) {
                lines.extend(split_lines);
                return;
            }
        }

        let direct_crop = crop_box(img, b);
        let mut direct = self.best_from_crop_direct(&direct_crop, cfg);
        let direct_is_strong = direct
            .as_ref()
            .is_some_and(|candidate| candidate.confidence >= MIN_STRONG_REC_CONFIDENCE);
        if allow_crop_enhancement
            && !direct_is_strong
            && *crop_enhancement_budget > 0
            && should_enhance_crop(b)
        {
            *crop_enhancement_budget -= 1;
            direct = self.best_from_crop(&direct_crop, cfg).or(direct);
        };

        if let Some(candidate) = direct {
            lines.push(TextLine {
                bbox: transform.map_box(b),
                text: normalize_recognized_text(&candidate.text),
                confidence: candidate.confidence,
                source: source.to_string(),
            });
        }
    }

    fn recognize_split_line_boxes(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        split_boxes: &[BoxRect],
        source: &str,
        transform: BboxTransform,
    ) -> Vec<TextLine> {
        let mut lines = Vec::new();
        for split_box in split_boxes {
            let crop = crop_box(img, *split_box);
            let candidate = self.best_from_crop_direct(&crop, cfg).or_else(|| {
                let binary = binarize_color_region_foreground(img, *split_box)?;
                self.recognize_best(&binary, cfg)
                    .ok()
                    .filter(is_usable_recognition)
            });
            if let Some(candidate) = candidate {
                lines.push(TextLine {
                    bbox: transform.map_box(*split_box),
                    text: normalize_recognized_text(&candidate.text),
                    confidence: candidate.confidence,
                    source: format!("{source}:split"),
                });
            }
        }
        lines
    }

    fn apply_quality_fallbacks(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        det_box_count: usize,
        text: &mut String,
        confidence: &mut f32,
        line_count: &mut usize,
        region_count: &mut usize,
        layout_applied: &mut bool,
        regions: &mut Vec<OcrTextRegion>,
        color_region_count: &mut usize,
        trace: &mut OcrTrace,
        fallback: &mut Option<String>,
    ) -> Result<(), String> {
        let image_bbox = image_box(img);
        trace.fallback_attempt_count += 1;
        match self.recognize_best(img, cfg) {
            Ok(candidate) if is_usable_recognition(&candidate) => {
                let label = recognition_fallback_label("whole-image", candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    label,
                    &candidate,
                );
            }
            Err(e) if text.trim().is_empty() => return Err(format!("recognize: {e}")),
            _ => {}
        }

        if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
            return Ok(());
        }

        trace.fallback_attempt_count += 1;
        let (candidate_count, candidate) = self.recognize_color_regions(img, cfg);
        *color_region_count = (*color_region_count).max(candidate_count);
        maybe_adopt_recognized(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            "color-regions".to_string(),
            &candidate,
        );
        if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
            return Ok(());
        }

        for (name, enhanced) in enhancement_variants(img) {
            trace.fallback_attempt_count += 1;
            trace.det_pass_count += 1;
            if let Ok(candidate) = self.recognize_detected_text(
                &enhanced,
                cfg,
                false,
                &format!("det-enhanced:{name}"),
                BboxTransform::Identity,
            ) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    format!("det-enhanced:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            trace.fallback_attempt_count += 1;
            if let Ok(candidate) = self.recognize_best(&enhanced, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("enhanced:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        let (orig_w, orig_h) = img.dimensions();
        for (name, upscaled) in upscale_variants(img) {
            trace.fallback_attempt_count += 1;
            trace.det_pass_count += 1;
            let (variant_w, variant_h) = upscaled.dimensions();
            let transform = BboxTransform::Scale {
                sx: orig_w as f32 / variant_w.max(1) as f32,
                sy: orig_h as f32 / variant_h.max(1) as f32,
                max_w: orig_w,
                max_h: orig_h,
            };
            if let Ok(candidate) = self.recognize_detected_text(
                &upscaled,
                cfg,
                false,
                &format!("det-upscaled:{name}"),
                transform,
            ) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    format!("det-upscaled:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            trace.fallback_attempt_count += 1;
            if let Ok(candidate) = self.recognize_best(&upscaled, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("upscaled:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        for (name, rotated) in rotation_variants(img) {
            trace.fallback_attempt_count += 1;
            trace.det_pass_count += 1;
            let transform = match name.as_str() {
                "90" => BboxTransform::Rotate90 {
                    src_w: orig_w,
                    src_h: orig_h,
                },
                "180" => BboxTransform::Rotate180 {
                    src_w: orig_w,
                    src_h: orig_h,
                },
                "270" => BboxTransform::Rotate270 {
                    src_w: orig_w,
                    src_h: orig_h,
                },
                _ => BboxTransform::Identity,
            };
            if let Ok(candidate) = self.recognize_detected_text(
                &rotated,
                cfg,
                false,
                &format!("det-rotated:{name}"),
                transform,
            ) {
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    format!("det-rotated:{name}"),
                    &candidate.recognized,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }

            trace.fallback_attempt_count += 1;
            if let Ok(candidate) = self.recognize_best(&rotated, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("rotated:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        trace.fallback_attempt_count += 1;
        let candidate = self.recognize_line_crops(img, cfg);
        maybe_adopt_recognized(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            "line-crops".to_string(),
            &candidate,
        );

        Ok(())
    }

    fn recognize_color_regions(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
    ) -> (usize, RecognizedText) {
        self.recognize_color_regions_limited(img, cfg, MAX_COLOR_REGION_CANDIDATES, "color-region")
    }

    fn recognize_color_regions_limited(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        recognition_limit: usize,
        source: &str,
    ) -> (usize, RecognizedText) {
        let boxes = color_region_boxes(img);
        let mut lines = Vec::new();
        for b in boxes.iter().take(recognition_limit) {
            let Some(binary) = binarize_color_region_foreground(img, *b) else {
                continue;
            };
            if let Ok(candidate) = self.recognize_best(&binary, cfg)
                && is_usable_recognition(&candidate)
            {
                lines.push(TextLine {
                    bbox: *b,
                    text: normalize_recognized_text(&candidate.text),
                    confidence: candidate.confidence,
                    source: source.to_string(),
                });
            }
        }
        (boxes.len(), recognized_from_text_lines(&mut lines))
    }

    fn recognize_line_crops(&self, img: &DynamicImage, cfg: &OcrConfig) -> RecognizedText {
        let mut lines = Vec::new();
        for line_box in fallback_line_boxes(img) {
            let line = crop_box(img, line_box);
            if let Ok(candidate) = self.recognize_best(&line, cfg)
                && is_usable_recognition(&candidate)
            {
                lines.push(TextLine {
                    bbox: line_box,
                    text: normalize_recognized_text(&candidate.text),
                    confidence: candidate.confidence,
                    source: "line-crops".to_string(),
                });
            }
        }
        recognized_from_text_lines(&mut lines)
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

fn recognized_from_text_lines(lines: &mut [TextLine]) -> RecognizedText {
    lines.sort_by(reading_line_order);
    let mut regions = group_text_lines_into_regions(lines);
    regions.sort_by(reading_region_order);

    let mut blocks = Vec::new();
    let mut confidence_sum = 0.0f32;
    let mut confidence_count = 0usize;
    let mut public_regions = Vec::new();
    for region in regions.iter_mut() {
        region.lines.sort_by(reading_line_order);
        let block = region
            .lines
            .iter()
            .map(|line| line.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if block.trim().is_empty() {
            continue;
        }
        for line in &region.lines {
            confidence_sum += line.confidence;
            confidence_count += 1;
        }
        public_regions.push(public_region_from_layout(region, &block));
        blocks.push(block);
    }

    let text = blocks.join("\n\n");
    let line_count = text_line_count(&text);
    let region_count = blocks.len();
    let confidence = if confidence_count == 0 {
        0.0
    } else {
        confidence_sum / confidence_count as f32
    };
    RecognizedText {
        text,
        confidence,
        line_count,
        region_count,
        layout_applied: region_count > 1,
        regions: public_regions,
    }
}

fn recognized_from_candidate(
    candidate: RecCandidate,
    bbox: BoxRect,
    source: &str,
) -> RecognizedText {
    let text = normalize_recognized_text(&candidate.text);
    let line_count = text_line_count(&text);
    let regions = if text.trim().is_empty() {
        Vec::new()
    } else {
        let line = OcrTextLine {
            bbox: box_to_array(bbox),
            text: text.clone(),
            confidence: candidate.confidence,
            source: source.to_string(),
        };
        vec![OcrTextRegion {
            bbox: box_to_array(bbox),
            text: text.clone(),
            confidence: candidate.confidence,
            source: source.to_string(),
            lines: vec![line],
        }]
    };
    RecognizedText {
        text,
        confidence: candidate.confidence,
        line_count,
        region_count: if line_count == 0 { 0 } else { 1 },
        layout_applied: false,
        regions,
    }
}

fn public_region_from_layout(region: &LayoutRegion, text: &str) -> OcrTextRegion {
    let lines = region
        .lines
        .iter()
        .map(|line| OcrTextLine {
            bbox: box_to_array(line.bbox),
            text: line.text.clone(),
            confidence: line.confidence,
            source: line.source.clone(),
        })
        .collect::<Vec<_>>();
    let confidence = if region.lines.is_empty() {
        0.0
    } else {
        region.lines.iter().map(|line| line.confidence).sum::<f32>() / region.lines.len() as f32
    };
    OcrTextRegion {
        bbox: box_to_array(region.bbox),
        text: text.to_string(),
        confidence,
        source: dominant_region_source(&region.lines),
        lines,
    }
}

fn dominant_region_source(lines: &[TextLine]) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for line in lines {
        if let Some((_, count)) = counts
            .iter_mut()
            .find(|(source, _)| *source == line.source.as_str())
        {
            *count += 1;
        } else {
            counts.push((line.source.as_str(), 1));
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(source, _)| source.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn merge_ocr_regions(current: &[OcrTextRegion], candidate: &[OcrTextRegion]) -> Vec<OcrTextRegion> {
    let mut out = current.to_vec();
    let mut seen = out
        .iter()
        .map(|region| normalize_ocr_line(&region.text))
        .collect::<Vec<_>>();
    for region in candidate {
        let key = normalize_ocr_line(&region.text);
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
        out.push(region.clone());
    }
    out
}

impl LayoutRegion {
    fn from_line(line: TextLine) -> Self {
        Self {
            bbox: line.bbox,
            lines: vec![line],
        }
    }

    fn add_line(&mut self, line: TextLine) {
        self.bbox = union_box(self.bbox, line.bbox);
        self.lines.push(line);
    }
}

fn group_text_lines_into_regions(lines: &[TextLine]) -> Vec<LayoutRegion> {
    let mut regions: Vec<LayoutRegion> = Vec::new();
    for line in lines.iter().filter(|line| !line.text.trim().is_empty()) {
        let mut best: Option<(usize, f32)> = None;
        for (idx, region) in regions.iter().enumerate() {
            if let Some(score) = region_line_score(region, line) {
                if best.map_or(true, |(_, best_score)| score > best_score) {
                    best = Some((idx, score));
                }
            }
        }

        if let Some((idx, _)) = best {
            regions[idx].add_line(line.clone());
        } else {
            regions.push(LayoutRegion::from_line(line.clone()));
        }
    }
    merge_layout_regions(regions)
}

fn merge_layout_regions(mut regions: Vec<LayoutRegion>) -> Vec<LayoutRegion> {
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                if regions_should_merge(&regions[i], &regions[j]) {
                    let other = regions.remove(j);
                    for line in other.lines {
                        regions[i].add_line(line);
                    }
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
    regions
}

fn region_line_score(region: &LayoutRegion, line: &TextLine) -> Option<f32> {
    let overlap = horizontal_overlap(region.bbox, line.bbox) as f32;
    let min_width = box_width(region.bbox).min(box_width(line.bbox)).max(1) as f32;
    let overlap_ratio = overlap / min_width;
    let y_gap = vertical_gap(region.bbox, line.bbox) as f32;
    let x_gap = horizontal_gap(region.bbox, line.bbox) as f32;
    let line_h = box_height(line.bbox).max(1) as f32;
    let avg_h = region_average_line_height(region).max(line_h);
    let same_row =
        vertical_overlap(region.bbox, line.bbox) > 0 && x_gap <= (line_h * 3.0).max(24.0);
    let max_y_gap = (avg_h * 4.0).clamp(48.0, 180.0);

    if !same_row && y_gap > max_y_gap {
        return None;
    }

    let region_w = box_width(region.bbox).max(1) as f32;
    let line_w = box_width(line.bbox).max(1) as f32;
    let width_ratio = region_w.max(line_w) / region_w.min(line_w);
    if !same_row && y_gap > avg_h * 2.5 && width_ratio >= 1.75 {
        return None;
    }

    let region_cx = box_center_x(region.bbox);
    let line_cx = box_center_x(line.bbox);
    let centers_close = (region_cx - line_cx).abs() <= min_width.max(64.0) * 0.55;
    if overlap_ratio < 0.25 && !same_row && !centers_close {
        return None;
    }

    Some(overlap_ratio * 100.0 + if same_row { 25.0 } else { 0.0 } - y_gap * 0.25 - x_gap * 0.02)
}

fn regions_should_merge(a: &LayoutRegion, b: &LayoutRegion) -> bool {
    let overlap = horizontal_overlap(a.bbox, b.bbox) as f32;
    let min_width = box_width(a.bbox).min(box_width(b.bbox)).max(1) as f32;
    let overlap_ratio = overlap / min_width;
    let y_gap = vertical_gap(a.bbox, b.bbox) as f32;
    let avg_h = region_average_line_height(a).max(region_average_line_height(b));
    let width_ratio = box_width(a.bbox).max(box_width(b.bbox)).max(1) as f32
        / box_width(a.bbox).min(box_width(b.bbox)).max(1) as f32;

    overlap_ratio >= 0.45 && y_gap <= (avg_h * 3.0).max(48.0) && width_ratio < 1.75
}

fn reading_line_order(a: &TextLine, b: &TextLine) -> std::cmp::Ordering {
    (a.bbox.1 / 8, a.bbox.0).cmp(&(b.bbox.1 / 8, b.bbox.0))
}

fn reading_region_order(a: &LayoutRegion, b: &LayoutRegion) -> std::cmp::Ordering {
    let y_close = vertical_overlap(a.bbox, b.bbox) > 0
        || a.bbox.1.abs_diff(b.bbox.1) <= (box_height(a.bbox).min(box_height(b.bbox)) / 3).max(24);
    if y_close {
        a.bbox
            .0
            .cmp(&b.bbox.0)
            .then_with(|| a.bbox.1.cmp(&b.bbox.1))
    } else {
        a.bbox
            .1
            .cmp(&b.bbox.1)
            .then_with(|| a.bbox.0.cmp(&b.bbox.0))
    }
}

fn union_box(a: BoxRect, b: BoxRect) -> BoxRect {
    (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
}

fn box_width(b: BoxRect) -> u32 {
    b.2.saturating_sub(b.0)
}

fn box_height(b: BoxRect) -> u32 {
    b.3.saturating_sub(b.1)
}

fn box_center_x(b: BoxRect) -> f32 {
    (b.0 as f32 + b.2 as f32) / 2.0
}

fn horizontal_overlap(a: BoxRect, b: BoxRect) -> u32 {
    a.2.min(b.2).saturating_sub(a.0.max(b.0))
}

fn vertical_overlap(a: BoxRect, b: BoxRect) -> u32 {
    a.3.min(b.3).saturating_sub(a.1.max(b.1))
}

fn horizontal_gap(a: BoxRect, b: BoxRect) -> u32 {
    if a.2 < b.0 {
        b.0 - a.2
    } else if b.2 < a.0 {
        a.0 - b.2
    } else {
        0
    }
}

fn vertical_gap(a: BoxRect, b: BoxRect) -> u32 {
    if a.3 < b.1 {
        b.1 - a.3
    } else if b.3 < a.1 {
        a.1 - b.3
    } else {
        0
    }
}

fn region_average_line_height(region: &LayoutRegion) -> f32 {
    if region.lines.is_empty() {
        return box_height(region.bbox).max(1) as f32;
    }
    region
        .lines
        .iter()
        .map(|line| box_height(line.bbox).max(1) as f32)
        .sum::<f32>()
        / region.lines.len() as f32
}

impl BboxTransform {
    fn map_box(self, b: BoxRect) -> BoxRect {
        match self {
            BboxTransform::Identity => b,
            BboxTransform::Scale {
                sx,
                sy,
                max_w,
                max_h,
            } => {
                let x0 = ((b.0 as f32) * sx).floor().max(0.0) as u32;
                let y0 = ((b.1 as f32) * sy).floor().max(0.0) as u32;
                let x1 = ((b.2 as f32) * sx).ceil().min(max_w as f32) as u32;
                let y1 = ((b.3 as f32) * sy).ceil().min(max_h as f32) as u32;
                clamp_box((x0, y0, x1, y1), max_w, max_h)
            }
            BboxTransform::Rotate90 { src_w, src_h } => {
                let mapped = (
                    b.1,
                    src_h.saturating_sub(b.2),
                    b.3,
                    src_h.saturating_sub(b.0),
                );
                clamp_box(mapped, src_w, src_h)
            }
            BboxTransform::Rotate180 { src_w, src_h } => {
                let mapped = (
                    src_w.saturating_sub(b.2),
                    src_h.saturating_sub(b.3),
                    src_w.saturating_sub(b.0),
                    src_h.saturating_sub(b.1),
                );
                clamp_box(mapped, src_w, src_h)
            }
            BboxTransform::Rotate270 { src_w, src_h } => {
                let mapped = (
                    src_w.saturating_sub(b.3),
                    b.0,
                    src_w.saturating_sub(b.1),
                    b.2,
                );
                clamp_box(mapped, src_w, src_h)
            }
        }
    }
}

fn clamp_box(b: BoxRect, max_w: u32, max_h: u32) -> BoxRect {
    let x0 = b.0.min(max_w);
    let y0 = b.1.min(max_h);
    let x1 = b.2.min(max_w).max(x0.saturating_add(1).min(max_w));
    let y1 = b.3.min(max_h).max(y0.saturating_add(1).min(max_h));
    (x0, y0, x1, y1)
}

fn image_box(image: &DynamicImage) -> BoxRect {
    let (w, h) = image.dimensions();
    (0, 0, w.max(1), h.max(1))
}

fn box_to_array(b: BoxRect) -> [u32; 4] {
    [b.0, b.1, b.2, b.3]
}

fn color_region_boxes(image: &DynamicImage) -> Vec<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (src_w, src_h) = rgb.dimensions();
    if src_w == 0 || src_h == 0 {
        return Vec::new();
    }

    let pixels = (src_w as u64).saturating_mul(src_h as u64);
    let (work, sx, sy) = if pixels > MAX_COLOR_REGION_PIXELS {
        let scale = (MAX_COLOR_REGION_PIXELS as f64 / pixels as f64).sqrt() as f32;
        let target_w = ((src_w as f32) * scale).round().max(1.0) as u32;
        let target_h = ((src_h as f32) * scale).round().max(1.0) as u32;
        let resized = image::imageops::resize(&rgb, target_w, target_h, FilterType::Triangle);
        (
            resized,
            src_w as f32 / target_w.max(1) as f32,
            src_h as f32 / target_h.max(1) as f32,
        )
    } else {
        (rgb, 1.0, 1.0)
    };

    let mut boxes = color_region_boxes_from_rgb(&work)
        .into_iter()
        .map(|b| {
            let x0 = ((b.0 as f32) * sx).floor().max(0.0) as u32;
            let y0 = ((b.1 as f32) * sy).floor().max(0.0) as u32;
            let x1 = ((b.2 as f32) * sx).ceil().min(src_w as f32) as u32;
            let y1 = ((b.3 as f32) * sy).ceil().min(src_h as f32) as u32;
            (x0, y0, x1.max(x0 + 1).min(src_w), y1.max(y0 + 1).min(src_h))
        })
        .collect::<Vec<_>>();
    boxes = nms_boxes(boxes, 0.70);
    boxes.truncate(MAX_COLOR_REGION_CANDIDATES);
    boxes
}

fn color_region_boxes_from_rgb(rgb: &image::RgbImage) -> Vec<BoxRect> {
    let (w_u32, h_u32) = rgb.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }

    let total = w.saturating_mul(h);
    let mut keys = vec![0u16; total];
    let mut counts = [0usize; 4096];
    for y in 0..h {
        for x in 0..w {
            let key = quantized_color_key(rgb.get_pixel(x as u32, y as u32));
            keys[y * w + x] = key;
            counts[key as usize] += 1;
        }
    }
    let dominant_key = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(key, _)| key as u16)
        .unwrap_or(0);

    let min_area = (total / 5000).clamp(24, 800);
    let mut visited = vec![false; total];
    let mut boxes = Vec::new();
    for idx in 0..total {
        if visited[idx] {
            continue;
        }
        let key = keys[idx];
        let mut stack = vec![idx];
        visited[idx] = true;
        let mut area = 0usize;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;

        while let Some(cur) = stack.pop() {
            let x = cur % w;
            let y = cur / w;
            area += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);

            if x > 0 {
                push_same_color_neighbor(cur - 1, key, &keys, &mut visited, &mut stack);
            }
            if x + 1 < w {
                push_same_color_neighbor(cur + 1, key, &keys, &mut visited, &mut stack);
            }
            if y > 0 {
                push_same_color_neighbor(cur - w, key, &keys, &mut visited, &mut stack);
            }
            if y + 1 < h {
                push_same_color_neighbor(cur + w, key, &keys, &mut visited, &mut stack);
            }
        }

        let rect_w = max_x.saturating_sub(min_x) + 1;
        let rect_h = max_y.saturating_sub(min_y) + 1;
        let bbox_area = rect_w.saturating_mul(rect_h);
        if area < min_area || rect_w < 24 || rect_h < 12 || bbox_area == 0 {
            continue;
        }
        if key == dominant_key && area.saturating_mul(100) > total.saturating_mul(15) {
            continue;
        }
        if bbox_area.saturating_mul(100) > total.saturating_mul(92) {
            continue;
        }
        let fill = area as f32 / bbox_area as f32;
        if fill < 0.55 {
            continue;
        }

        boxes.push(expand_color_region_box(min_x, min_y, max_x, max_y, w, h));
    }

    boxes.sort_by_key(|b| (b.1, b.0));
    boxes
}

fn push_same_color_neighbor(
    idx: usize,
    key: u16,
    keys: &[u16],
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    if !visited[idx] && keys[idx] == key {
        visited[idx] = true;
        stack.push(idx);
    }
}

fn expand_color_region_box(
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    w: usize,
    h: usize,
) -> BoxRect {
    let pad = 2usize;
    (
        min_x.saturating_sub(pad) as u32,
        min_y.saturating_sub(pad) as u32,
        (max_x + 1 + pad).min(w) as u32,
        (max_y + 1 + pad).min(h) as u32,
    )
}

fn quantized_color_key(pixel: &image::Rgb<u8>) -> u16 {
    let r = (pixel[0] >> 4) as u16;
    let g = (pixel[1] >> 4) as u16;
    let b = (pixel[2] >> 4) as u16;
    (r << 8) | (g << 4) | b
}

fn binarize_color_region_foreground(image: &DynamicImage, b: BoxRect) -> Option<DynamicImage> {
    let crop = crop_box(image, b);
    let rgb = to_rgb_on_white(&crop);
    let (w, h) = rgb.dimensions();
    if w < 8 || h < 6 {
        return None;
    }

    let bg = estimate_region_background_rgb(&rgb);
    let mut distances = Vec::with_capacity((w as usize).saturating_mul(h as usize));
    let mut max_distance = 0u8;
    for pixel in rgb.pixels() {
        let distance = color_distance_u8(pixel, bg);
        max_distance = max_distance.max(distance);
        distances.push(distance);
    }
    if max_distance < 18 {
        return None;
    }

    let threshold = otsu_threshold_values(&distances).clamp(18, 96);
    let mut foreground_count = 0usize;
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = (y as usize) * w as usize + x as usize;
            let foreground = distances[idx] >= threshold;
            if foreground {
                foreground_count += 1;
            }
            out.put_pixel(x, y, Luma([if foreground { 0 } else { 255 }]));
        }
    }

    let total = (w as usize).saturating_mul(h as usize).max(1);
    let foreground_ratio = foreground_count as f32 / total as f32;
    if foreground_count < 4 || !(0.002..=0.45).contains(&foreground_ratio) {
        return None;
    }
    Some(DynamicImage::ImageLuma8(out))
}

fn estimate_region_background_rgb(rgb: &image::RgbImage) -> [u8; 3] {
    let (w, h) = rgb.dimensions();
    let mut counts = [0usize; 4096];
    let mut sums = [[0u64; 3]; 4096];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x, y);
            let key = quantized_color_key(p) as usize;
            counts[key] += 1;
            sums[key][0] += p[0] as u64;
            sums[key][1] += p[1] as u64;
            sums[key][2] += p[2] as u64;
        }
    }

    let key = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(key, _)| key)
        .unwrap_or(0);
    let count = counts[key].max(1) as u64;
    [
        (sums[key][0] / count) as u8,
        (sums[key][1] / count) as u8,
        (sums[key][2] / count) as u8,
    ]
}

fn color_distance_u8(pixel: &image::Rgb<u8>, bg: [u8; 3]) -> u8 {
    let dr = pixel[0].abs_diff(bg[0]);
    let dg = pixel[1].abs_diff(bg[1]);
    let db = pixel[2].abs_diff(bg[2]);
    dr.max(dg).max(db)
}

fn otsu_threshold_values(values: &[u8]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let mut hist = [0u64; 256];
    for value in values {
        hist[*value as usize] += 1;
    }
    otsu_threshold_histogram(&hist, values.len() as u64)
}

fn otsu_threshold_histogram(hist: &[u64; 256], total: u64) -> u8 {
    if total == 0 {
        return 0;
    }
    let mut sum_total = 0u64;
    for (value, count) in hist.iter().enumerate() {
        sum_total += value as u64 * count;
    }

    let mut weight_bg = 0u64;
    let mut sum_bg = 0u64;
    let mut best_threshold = 0u8;
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
    region_count: &mut usize,
    layout_applied: &mut bool,
    regions: &mut Vec<OcrTextRegion>,
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
        *region_count = candidate
            .region_count
            .max(if text.trim().is_empty() { 0 } else { 1 });
        *layout_applied = candidate.layout_applied;
        *regions = candidate.regions.clone();
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
        *region_count = candidate
            .region_count
            .max(if text.trim().is_empty() { 0 } else { 1 });
        *layout_applied = candidate.layout_applied;
        *regions = candidate.regions.clone();
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
        *region_count = (*region_count).max(candidate.region_count).max(1);
        *layout_applied = *layout_applied || candidate.layout_applied || *region_count > 1;
        *regions = merge_ocr_regions(regions, &candidate.regions);
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
        *region_count = candidate
            .region_count
            .max(if text.trim().is_empty() { 0 } else { 1 });
        *layout_applied = candidate.layout_applied;
        *regions = candidate.regions.clone();
        *fallback = Some(label);
        return true;
    }

    false
}

fn filter_non_overlapping_recognized(
    candidate: &RecognizedText,
    existing_regions: &[OcrTextRegion],
) -> RecognizedText {
    let existing_boxes = collect_region_line_boxes(existing_regions);
    if existing_boxes.is_empty() {
        return candidate.clone();
    }

    let mut lines = Vec::new();
    for region in &candidate.regions {
        for line in &region.lines {
            let bbox = box_from_array(line.bbox);
            if existing_boxes
                .iter()
                .any(|existing| boxes_significantly_overlap(bbox, *existing))
            {
                continue;
            }
            lines.push(TextLine {
                bbox,
                text: line.text.clone(),
                confidence: line.confidence,
                source: line.source.clone(),
            });
        }
    }

    recognized_from_text_lines(&mut lines)
}

fn collect_region_line_boxes(regions: &[OcrTextRegion]) -> Vec<BoxRect> {
    let mut boxes = Vec::new();
    for region in regions {
        if region.lines.is_empty() {
            boxes.push(box_from_array(region.bbox));
            continue;
        }
        for line in &region.lines {
            boxes.push(box_from_array(line.bbox));
        }
    }
    boxes
}

fn box_from_array(b: [u32; 4]) -> BoxRect {
    (b[0], b[1], b[2], b[3])
}

fn boxes_significantly_overlap(a: BoxRect, b: BoxRect) -> bool {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    if x1 <= x0 || y1 <= y0 {
        return false;
    }
    let area = box_area((x0, y0, x1, y1));
    let min_area = box_area(a).min(box_area(b)).max(1);
    area as f32 / min_area as f32 >= 0.50
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

fn normalize_recognized_text(text: &str) -> String {
    split_joined_chat_time_markers(text)
}

fn split_joined_chat_time_markers(text: &str) -> String {
    if text.contains('\n') {
        return text.to_string();
    }
    let markers = [
        "刚刚",
        "昨天",
        "星期一",
        "星期二",
        "星期三",
        "星期四",
        "星期五",
        "星期六",
        "星期日",
        "星期天",
    ];
    for marker in markers {
        let Some(idx) = text.find(marker) else {
            continue;
        };
        if idx == 0 {
            continue;
        }
        let before = &text[..idx];
        let after = &text[idx..];
        let before_chars = recognized_char_count(before);
        let after_chars = recognized_char_count(after);
        let has_sender_prefix = before.contains('：') || before.contains(':');
        if has_sender_prefix && before_chars >= 6 && after_chars >= 4 {
            return format!("{}\n{}", before.trim(), after.trim());
        }
    }
    text.to_string()
}

fn should_use_split_lines(direct: Option<&RecCandidate>, split_lines: &[TextLine]) -> bool {
    if split_lines.len() < 2 {
        return false;
    }

    let split_text = split_lines
        .iter()
        .map(|line| line.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let split_chars = recognized_char_count(&split_text);
    if split_chars < 4 {
        return false;
    }

    let split_confidence = split_lines.iter().map(|line| line.confidence).sum::<f32>()
        / split_lines.len().max(1) as f32;

    let Some(direct) = direct else {
        return true;
    };

    let direct_chars = recognized_char_count(&direct.text);
    if direct_chars == 0 {
        return true;
    }

    let keeps_most_content =
        split_chars + 2 >= direct_chars || split_chars as f32 >= direct_chars as f32 * 0.72;
    let confidence_is_close = split_confidence + 0.15 >= direct.confidence
        || split_confidence >= MIN_STRONG_REC_CONFIDENCE;
    keeps_most_content && confidence_is_close
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
        let text_is_strong = confidence >= MIN_STRONG_REC_CONFIDENCE
            && char_count >= 8
            && readable_ratio(text) >= 0.70;
        return !text_is_strong;
    }
    if char_count < 4 && det_box_count >= 2 {
        return true;
    }
    if char_count >= 4 && readable_ratio(text) < 0.55 {
        return true;
    }
    false
}

fn should_enhance_crop(b: BoxRect) -> bool {
    box_width(b) <= 480 && box_height(b) <= 96 && box_area(b) <= 48_000
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

fn ocr_trace_enabled() -> bool {
    std::env::var("VECTRAPARSE_OCR_TRACE").ok().as_deref() == Some("1")
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

fn fallback_line_boxes(image: &DynamicImage) -> Vec<BoxRect> {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    foreground_line_boxes(image, 48)
}

fn split_text_box_into_line_boxes(image: &DynamicImage, bbox: BoxRect) -> Vec<BoxRect> {
    if box_height(bbox) < 18 || box_width(bbox) < 16 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let local_boxes = foreground_line_boxes(&crop, 8);
    if local_boxes.is_empty() {
        return Vec::new();
    }

    let mut split_boxes = Vec::new();
    for line_box in local_boxes {
        let segments = split_line_box_horizontally(&crop, line_box);
        if segments.len() > 1 {
            split_boxes.extend(segments);
        } else {
            split_boxes.push(line_box);
        }
    }
    if split_boxes.len() < 2 {
        return Vec::new();
    }

    let (img_w, img_h) = image.dimensions();
    split_boxes
        .into_iter()
        .map(|b| {
            clamp_box(
                (
                    bbox.0.saturating_add(b.0),
                    bbox.1.saturating_add(b.1),
                    bbox.0.saturating_add(b.2),
                    bbox.1.saturating_add(b.3),
                ),
                img_w,
                img_h,
            )
        })
        .collect()
}

fn split_text_box_into_color_region_boxes(image: &DynamicImage, bbox: BoxRect) -> Vec<BoxRect> {
    if box_width(bbox) < 96 || box_height(bbox) < 24 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let crop_area = box_area(image_box(&crop));
    let mut local_boxes = color_region_boxes(&crop)
        .into_iter()
        .filter(|b| box_width(*b) >= 24 && box_height(*b) >= 12)
        .filter(|b| box_area(*b).saturating_mul(100) < crop_area.saturating_mul(88))
        .collect::<Vec<_>>();
    if local_boxes.len() == 1
        && let Some(foreground_box) = foreground_box_outside_boxes(&crop, &local_boxes)
    {
        local_boxes.push(foreground_box);
    }

    let mut boxes = local_boxes
        .into_iter()
        .map(|b| {
            clamp_box(
                (
                    bbox.0.saturating_add(b.0),
                    bbox.1.saturating_add(b.1),
                    bbox.0.saturating_add(b.2),
                    bbox.1.saturating_add(b.3),
                ),
                image.width(),
                image.height(),
            )
        })
        .collect::<Vec<_>>();
    boxes.sort_by_key(|b| (b.1 / 8, b.0));
    boxes.truncate(MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS);
    if boxes.len() < 2 {
        return Vec::new();
    }
    boxes
}

fn foreground_box_outside_boxes(image: &DynamicImage, excluded: &[BoxRect]) -> Option<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    let mask = foreground_mask_from_rgb(&rgb).or_else(|| dark_luma_mask_from_rgb(&rgb))?;

    let mut min_x = w as usize;
    let mut min_y = h as usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut count = 0usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            if !mask[y * w as usize + x]
                || excluded
                    .iter()
                    .any(|b| point_in_box(x as u32, y as u32, *b))
            {
                continue;
            }
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if count < 4 || max_x <= min_x || max_y <= min_y {
        return None;
    }
    let b = (
        min_x.saturating_sub(2) as u32,
        min_y.saturating_sub(2) as u32,
        (max_x + 3).min(w as usize) as u32,
        (max_y + 3).min(h as usize) as u32,
    );
    if box_width(b) < 24 || box_height(b) < 8 {
        return None;
    }
    Some(b)
}

fn point_in_box(x: u32, y: u32, b: BoxRect) -> bool {
    x >= b.0 && x < b.2 && y >= b.1 && y < b.3
}

fn split_line_box_horizontally(image: &DynamicImage, bbox: BoxRect) -> Vec<BoxRect> {
    if box_width(bbox) < 64 || box_height(bbox) < 6 {
        return vec![bbox];
    }

    let crop = crop_box(image, bbox);
    let rgb = to_rgb_on_white(&crop);
    let (w, h) = rgb.dimensions();
    let Some(mask) = foreground_mask_from_rgb(&rgb).or_else(|| dark_luma_mask_from_rgb(&rgb))
    else {
        return vec![bbox];
    };
    let local_segments = column_boxes_from_foreground_mask(
        &mask,
        w as usize,
        h as usize,
        MAX_HORIZONTAL_SEGMENTS_PER_LINE,
    );
    if local_segments.len() < 2 {
        return vec![bbox];
    }

    local_segments
        .into_iter()
        .map(|b| {
            clamp_box(
                (
                    bbox.0.saturating_add(b.0),
                    bbox.1.saturating_add(b.1),
                    bbox.0.saturating_add(b.2),
                    bbox.1.saturating_add(b.3),
                ),
                image.width(),
                image.height(),
            )
        })
        .collect()
}

fn foreground_line_boxes(image: &DynamicImage, max_boxes: usize) -> Vec<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(mask) = foreground_mask_from_rgb(&rgb).or_else(|| dark_luma_mask_from_rgb(&rgb))
    else {
        return Vec::new();
    };
    line_boxes_from_foreground_mask(&mask, w as usize, h as usize, max_boxes)
}

fn foreground_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return None;
    }

    let bg = estimate_region_background_rgb(rgb);
    let mut distances = Vec::with_capacity((w as usize).saturating_mul(h as usize));
    let mut max_distance = 0u8;
    for pixel in rgb.pixels() {
        let distance = color_distance_u8(pixel, bg);
        max_distance = max_distance.max(distance);
        distances.push(distance);
    }
    if max_distance < 14 {
        return None;
    }

    let threshold = otsu_threshold_values(&distances).clamp(12, 96);
    let mut foreground_count = 0usize;
    let mask = distances
        .into_iter()
        .map(|distance| {
            let foreground = distance >= threshold;
            if foreground {
                foreground_count += 1;
            }
            foreground
        })
        .collect::<Vec<_>>();

    let total = (w as usize).saturating_mul(h as usize).max(1);
    let foreground_ratio = foreground_count as f32 / total as f32;
    if foreground_count < 4 || !(0.001..=0.50).contains(&foreground_ratio) {
        return None;
    }
    Some(mask)
}

fn dark_luma_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return None;
    }

    let mut foreground_count = 0usize;
    let mut mask = Vec::with_capacity((w as usize).saturating_mul(h as usize));
    for pixel in rgb.pixels() {
        let lum = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
        let foreground = lum < 230;
        if foreground {
            foreground_count += 1;
        }
        mask.push(foreground);
    }

    let total = (w as usize).saturating_mul(h as usize).max(1);
    let foreground_ratio = foreground_count as f32 / total as f32;
    if foreground_count < 4 || !(0.001..=0.65).contains(&foreground_ratio) {
        return None;
    }
    Some(mask)
}

fn line_boxes_from_foreground_mask(
    mask: &[bool],
    w: usize,
    h: usize,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return Vec::new();
    }

    let mut row_score = vec![0usize; h];
    for y in 0..h {
        let mut count = 0usize;
        for x in 0..w {
            if mask[y * w + x] {
                count += 1;
            }
        }
        row_score[y] = count;
    }

    let max_row_score = row_score.iter().copied().max().unwrap_or(0);
    if max_row_score < 2 {
        return Vec::new();
    }
    let active_threshold = ((max_row_score as f32) * 0.10).ceil() as usize;
    let active_threshold = active_threshold.max((w / 180).max(2));
    let bridge_threshold = (active_threshold / 2).max(1);
    let gap_tolerance = (h / 80).clamp(1, 3);
    let min_band_height = (h / 120).clamp(3, 8);

    let mut bands = Vec::new();
    let mut y = 0usize;
    while y < h {
        if row_score[y] < active_threshold {
            y += 1;
            continue;
        }
        let start = y;
        let mut end = y;
        let mut gap = 0usize;
        y += 1;
        while y < h {
            if row_score[y] >= bridge_threshold {
                end = y;
                gap = 0;
            } else {
                gap += 1;
                if gap > gap_tolerance {
                    break;
                }
            }
            y += 1;
        }
        let height = end - start + 1;
        if height < min_band_height {
            continue;
        }
        bands.push((start, end));
    }

    let mut boxes = Vec::new();
    for (start, end) in bands {
        let mut min_x = w;
        let mut max_x = 0usize;
        let mut foreground_count = 0usize;
        for yy in start..=end {
            for xx in 0..w {
                if mask[yy * w + xx] {
                    foreground_count += 1;
                    min_x = min_x.min(xx);
                    max_x = max_x.max(xx);
                }
            }
        }
        if max_x <= min_x || foreground_count < 4 || max_x - min_x + 1 < 8 {
            continue;
        }
        let band_h = end - start + 1;
        let x_pad = (band_h / 2).clamp(2, 8);
        let y_pad = (band_h / 4).clamp(1, 3);
        boxes.push((
            min_x.saturating_sub(x_pad) as u32,
            start.saturating_sub(y_pad) as u32,
            (max_x + 1 + x_pad).min(w) as u32,
            (end + 1 + y_pad).min(h) as u32,
        ));
    }

    boxes.sort_by_key(|b| (b.1, b.0));
    boxes.truncate(max_boxes);
    boxes
}

fn column_boxes_from_foreground_mask(
    mask: &[bool],
    w: usize,
    h: usize,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return Vec::new();
    }

    let mut col_score = vec![0usize; w];
    for x in 0..w {
        let mut count = 0usize;
        for y in 0..h {
            if mask[y * w + x] {
                count += 1;
            }
        }
        col_score[x] = count;
    }

    let max_col_score = col_score.iter().copied().max().unwrap_or(0);
    if max_col_score < 2 {
        return Vec::new();
    }
    let active_threshold = ((max_col_score as f32) * 0.08).ceil().max(1.0) as usize;
    let bridge_threshold = (active_threshold / 2).max(1);
    let gap_tolerance = h.clamp(12, 32);
    let min_segment_width = h.clamp(8, 24);

    let mut bands = Vec::new();
    let mut x = 0usize;
    while x < w {
        if col_score[x] < active_threshold {
            x += 1;
            continue;
        }
        let start = x;
        let mut end = x;
        let mut gap = 0usize;
        x += 1;
        while x < w {
            if col_score[x] >= bridge_threshold {
                end = x;
                gap = 0;
            } else {
                gap += 1;
                if gap > gap_tolerance {
                    break;
                }
            }
            x += 1;
        }
        if end.saturating_sub(start) + 1 >= min_segment_width {
            bands.push((start, end));
        }
    }
    if bands.len() < 2 || bands.len() > max_boxes {
        return Vec::new();
    }

    let mut boxes = Vec::new();
    for (start, end) in bands {
        let mut min_y = h;
        let mut max_y = 0usize;
        let mut foreground_count = 0usize;
        for yy in 0..h {
            for xx in start..=end {
                if mask[yy * w + xx] {
                    foreground_count += 1;
                    min_y = min_y.min(yy);
                    max_y = max_y.max(yy);
                }
            }
        }
        if max_y <= min_y || foreground_count < 4 {
            continue;
        }
        let y_pad = ((max_y - min_y + 1) / 4).clamp(1, 3);
        let x_pad = h.clamp(2, 8) / 2;
        boxes.push((
            start.saturating_sub(x_pad) as u32,
            min_y.saturating_sub(y_pad) as u32,
            (end + 1 + x_pad).min(w) as u32,
            (max_y + 1 + y_pad).min(h) as u32,
        ));
    }

    boxes.sort_by_key(|b| (b.0, b.1));
    boxes
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
        assert_eq!(dynamic_rec_target_width(&long, 48, 320), 640);
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
        assert!(!needs_quality_fallback("Readable line", 0.62, 6, 1));
        assert!(!needs_quality_fallback("Invoice 42", 0.62, 2, 2));
    }

    #[test]
    fn normalize_recognized_text_splits_joined_chat_time_marker() {
        assert_eq!(
            normalize_recognized_text("陈晗：mac是用内核导的刚刚网关说不支持邮件"),
            "陈晗：mac是用内核导的\n刚刚网关说不支持邮件"
        );
        assert_eq!(
            normalize_recognized_text("那可能邮件这块的时间有点兜不住"),
            "那可能邮件这块的时间有点兜不住"
        );
    }

    #[test]
    fn maybe_adopt_recognized_merges_unique_lines() {
        let mut text = "Header".to_string();
        let mut confidence = 0.44;
        let mut line_count = 1;
        let mut region_count = 1;
        let mut layout_applied = false;
        let mut regions = Vec::new();
        let mut fallback = None;
        let candidate = RecognizedText {
            text: "Header\nTotal 42".to_string(),
            confidence: 0.54,
            line_count: 2,
            region_count: 1,
            layout_applied: false,
            regions: vec![OcrTextRegion {
                bbox: [0, 0, 120, 32],
                text: "Header\nTotal 42".to_string(),
                confidence: 0.54,
                source: "det-enhanced:contrast".to_string(),
                lines: Vec::new(),
            }],
        };
        assert!(maybe_adopt_recognized(
            &mut text,
            &mut confidence,
            &mut line_count,
            &mut region_count,
            &mut layout_applied,
            &mut regions,
            &mut fallback,
            "det-enhanced:contrast".to_string(),
            &candidate,
        ));
        assert_eq!(text, "Header\nTotal 42");
        assert_eq!(line_count, 2);
        assert_eq!(region_count, 1);
        assert!(!layout_applied);
        assert_eq!(regions.len(), 1);
        assert_eq!(fallback.as_deref(), Some("merged:det-enhanced:contrast"));
    }

    #[test]
    fn eager_color_regions_skip_existing_text_boxes() {
        let existing_regions = vec![OcrTextRegion {
            bbox: [10, 10, 70, 26],
            text: "Header".to_string(),
            confidence: 0.78,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 70, 26],
                text: "Header".to_string(),
                confidence: 0.78,
                source: "det".to_string(),
            }],
        }];
        let mut candidate_lines = vec![
            text_line((6, 6, 76, 30), "Header noise", 0.64),
            text_line((120, 10, 180, 26), "New item", 0.66),
        ];
        let candidate = recognized_from_text_lines(&mut candidate_lines);
        let filtered = filter_non_overlapping_recognized(&candidate, &existing_regions);

        assert_eq!(filtered.text, "New item");
        assert_eq!(filtered.line_count, 1);
        assert_eq!(filtered.regions.len(), 1);
        assert_eq!(filtered.regions[0].bbox, [120, 10, 180, 26]);
    }

    #[test]
    fn layout_regions_keep_columns_separate() {
        let mut lines = vec![
            text_line((0, 0, 90, 12), "Left A", 0.80),
            text_line((0, 18, 90, 30), "Left B", 0.82),
            text_line((220, 0, 330, 12), "Right A", 0.81),
            text_line((220, 18, 330, 30), "Right B", 0.83),
        ];
        let recognized = recognized_from_text_lines(&mut lines);
        assert_eq!(recognized.text, "Left A\nLeft B\n\nRight A\nRight B");
        assert_eq!(recognized.line_count, 4);
        assert_eq!(recognized.region_count, 2);
        assert!(recognized.layout_applied);
        assert_eq!(recognized.regions[0].bbox, [0, 0, 90, 30]);
        assert_eq!(recognized.regions[0].lines[0].source, "det");
    }

    #[test]
    fn layout_regions_do_not_merge_full_width_header_with_columns() {
        let mut lines = vec![
            text_line((0, 0, 320, 14), "Header", 0.90),
            text_line((0, 56, 90, 68), "Menu", 0.80),
            text_line((150, 56, 320, 68), "Content", 0.82),
        ];
        let recognized = recognized_from_text_lines(&mut lines);
        assert_eq!(recognized.text, "Header\n\nMenu\n\nContent");
        assert_eq!(recognized.region_count, 3);
        assert!(recognized.layout_applied);
    }

    #[test]
    fn color_region_boxes_detects_contrasting_panel() {
        let mut rgb = image::RgbImage::from_pixel(160, 80, image::Rgb([248, 248, 248]));
        for y in 20..50 {
            for x in 30..130 {
                rgb.put_pixel(x, y, image::Rgb([32, 88, 180]));
            }
        }
        let boxes = color_region_boxes(&DynamicImage::ImageRgb8(rgb));
        assert!(boxes.iter().any(|b| {
            b.0 <= 30 && b.1 <= 20 && b.2 >= 130 && b.3 >= 50 && box_area(*b) < 160 * 80
        }));
    }

    #[test]
    fn color_region_boxes_detects_subtle_light_panel() {
        let mut rgb = image::RgbImage::from_pixel(160, 80, image::Rgb([246, 247, 249]));
        for y in 20..50 {
            for x in 30..130 {
                rgb.put_pixel(x, y, image::Rgb([228, 231, 235]));
            }
        }
        for y in 31..37 {
            for x in 48..112 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let boxes = color_region_boxes(&DynamicImage::ImageRgb8(rgb));
        assert!(boxes.iter().any(|b| {
            b.0 <= 30 && b.1 <= 20 && b.2 >= 130 && b.3 >= 50 && box_area(*b) < 160 * 80
        }));
    }

    #[test]
    fn color_region_binarization_handles_padded_light_panel() {
        let mut rgb = image::RgbImage::from_pixel(160, 80, image::Rgb([246, 247, 249]));
        for y in 20..50 {
            for x in 30..130 {
                rgb.put_pixel(x, y, image::Rgb([228, 231, 235]));
            }
        }
        for y in 31..37 {
            for x in 48..112 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = color_region_boxes(&DynamicImage::ImageRgb8(rgb.clone()));
        let panel = boxes
            .into_iter()
            .find(|b| b.0 <= 30 && b.1 <= 20 && b.2 >= 130 && b.3 >= 50)
            .expect("light panel candidate");
        let binary = binarize_color_region_foreground(&DynamicImage::ImageRgb8(rgb), panel)
            .expect("binary region");
        let gray = binary.to_luma8();
        let panel_bg_x = 35u32.saturating_sub(panel.0);
        let panel_bg_y = 24u32.saturating_sub(panel.1);
        let text_x = 64u32.saturating_sub(panel.0);
        let text_y = 33u32.saturating_sub(panel.1);

        assert_eq!(gray.get_pixel(panel_bg_x, panel_bg_y)[0], 255);
        assert_eq!(gray.get_pixel(text_x, text_y)[0], 0);
    }

    #[test]
    fn color_region_binarization_extracts_foreground_from_panel() {
        let mut rgb = image::RgbImage::from_pixel(64, 24, image::Rgb([32, 88, 180]));
        for y in 8..14 {
            for x in 18..46 {
                rgb.put_pixel(x, y, image::Rgb([246, 246, 246]));
            }
        }
        let binary =
            binarize_color_region_foreground(&DynamicImage::ImageRgb8(rgb), (0, 0, 64, 24))
                .expect("binary region");
        let gray = binary.to_luma8();
        assert_eq!(gray.get_pixel(2, 2)[0], 255);
        assert_eq!(gray.get_pixel(24, 10)[0], 0);
    }

    #[test]
    fn split_text_box_into_line_boxes_splits_two_rows() {
        let mut rgb = image::RgbImage::from_pixel(96, 48, image::Rgb([255, 255, 255]));
        for y in 8..14 {
            for x in 12..70 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        for y in 30..36 {
            for x in 10..84 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = split_text_box_into_line_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 96, 48));
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].1 <= 8 && boxes[0].3 >= 14);
        assert!(boxes[1].1 <= 30 && boxes[1].3 >= 36);
    }

    #[test]
    fn split_text_box_into_line_boxes_ignores_single_row() {
        let mut rgb = image::RgbImage::from_pixel(96, 32, image::Rgb([255, 255, 255]));
        for y in 12..18 {
            for x in 12..84 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = split_text_box_into_line_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 96, 32));
        assert!(boxes.is_empty());
    }

    #[test]
    fn split_text_box_into_line_boxes_splits_wide_row_on_large_gap() {
        let mut rgb = image::RgbImage::from_pixel(180, 32, image::Rgb([255, 255, 255]));
        for y in 12..18 {
            for x in 12..62 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
            for x in 112..166 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = split_text_box_into_line_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 180, 32));
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].2 < boxes[1].0);
    }

    #[test]
    fn split_text_box_into_color_region_boxes_splits_adjacent_panels() {
        let mut rgb = image::RgbImage::from_pixel(180, 48, image::Rgb([245, 247, 250]));
        for y in 6..42 {
            for x in 8..82 {
                rgb.put_pixel(x, y, image::Rgb([50, 130, 238]));
            }
        }
        for y in 10..38 {
            for x in 96..170 {
                rgb.put_pixel(x, y, image::Rgb([230, 232, 236]));
            }
        }
        for y in 20..25 {
            for x in 110..154 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes =
            split_text_box_into_color_region_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 180, 48));
        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].2 <= boxes[1].0);
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

    fn text_line(bbox: BoxRect, text: &str, confidence: f32) -> TextLine {
        TextLine {
            bbox,
            text: text.to_string(),
            confidence,
            source: "det".to_string(),
        }
    }
}
