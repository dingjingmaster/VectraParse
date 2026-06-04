use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

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
const MIN_SUPPLEMENT_CHAR_GROWTH: usize = 3;
const MIN_SUPPLEMENT_CONFIDENCE_GAIN: f32 = 0.03;
const MAX_EAGER_COLOR_REGION_RECOGNITIONS: usize = 8;
const MAX_EAGER_COLOR_REGION_DET_PASSES: usize = 4;
const MAX_EAGER_LAYERED_REGION_RECOGNITIONS: usize = 10;
const MAX_EAGER_VISUAL_REGION_RECOGNITIONS: usize = 6;
const MAX_EAGER_SUPPLEMENT_RECOGNITIONS_TOTAL: usize = 12;
const MAX_EAGER_SUPPLEMENT_DET_PASSES_TOTAL: usize = 4;
const MAX_SUPPLEMENT_OUTSIDE_FOCUS_CANDIDATES: usize = 3;
const MAX_QUALITY_FALLBACK_FAMILIES_EMPTY: usize = 5;
const MAX_QUALITY_FALLBACK_FAMILIES_PARTIAL: usize = 3;
const MAX_REC_IMG_W: usize = 960;
const MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS: usize = 4;
const MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS: usize = 6;
const MAX_LINE_REPAIR_RECOGNITIONS_PER_PASS: usize = 8;
const MAX_PAGE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS: usize = 4;
const MAX_PAGE_REGION_REPAIR_RECOGNITIONS_PER_PASS: usize = 4;
const MAX_COLOR_REGION_DET_SPLIT_LINE_RECOGNITIONS_PER_PASS: usize = 3;
const MAX_COLOR_REGION_DET_REPAIR_RECOGNITIONS_PER_PASS: usize = 3;
const MAX_TILE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS: usize = 4;
const MAX_TILE_REGION_REPAIR_RECOGNITIONS_PER_PASS: usize = 4;
const MAX_QUALITY_FALLBACK_ENHANCEMENT_VARIANTS: usize = 6;
const MAX_HIGH_RES_TILE_DET_PASSES: usize = 8;
const MAX_HORIZONTAL_SEGMENTS_PER_LINE: usize = 4;
const MAX_WIDE_LINE_SEGMENTS_PER_LINE: usize = 6;
const MAX_RAW_DET_SPLIT_CANDIDATES: usize = 12;
const MAX_SUPPLEMENT_CANDIDATE_SCORE_PRESELECT: usize = 96;
const MAX_PAGE_REGION_DET_PASSES: usize = 4;
const MAX_PANEL_CHILD_CANDIDATES: usize = 6;
const MAX_PANEL_RECURSION_DEPTH: usize = 2;
const MAX_LOCAL_DET_UPSCALE_PASSES_PER_REGION: usize = 1;
const MAX_ENHANCEMENT_VARIANTS_PER_PASS: usize = 4;
const DETECTION_CONTAINMENT_OVERLAP: f32 = 0.88;
const DETECTION_SIMILARITY_OVERLAP: f32 = 0.90;
const GRAPH_REGION_VERTICAL_GAP_LIMIT: u32 = 96;
const CTC_BEAM_SIZE: usize = 4;
const CTC_TOP_K: usize = 4;
const MIN_ACCEPT_REC_CONFIDENCE: f32 = 0.25;
const MIN_STRONG_REC_CONFIDENCE: f32 = 0.55;
const MAX_REC_IMAGE_SIGNATURE_CACHE_ENTRIES: usize = 1024;
const MAX_REC_CANDIDATE_CACHE_ENTRIES: usize = 512;
const MAX_REC_PREPROCESS_CACHE_ENTRIES: usize = 768;
const MAX_REC_PREPARED_IMAGE_CACHE_ENTRIES: usize = 192;
const MAX_RGB_IMAGE_CACHE_ENTRIES: usize = 256;
const MAX_LUMA_IMAGE_CACHE_ENTRIES: usize = 256;
const BOX_DEDUPE_BUCKET_SIZE: u32 = 64;
const LOCAL_RECOGNITION_TINY_CROP_AREA: u64 = 3_000;
const LOCAL_RECOGNITION_SMALL_CROP_AREA: u64 = 10_000;
const MAX_FAST_TEXT_CHARS: usize = 10;
const CROP_ENHANCE_TINY_AREA: u64 = 4_000;
const CROP_ENHANCE_SMALL_AREA: u64 = 8_000;
const STABLE_TEXT_CONFIDENCE: f32 = 0.90;
const STABLE_TEXT_AVG_MARGIN: f32 = 0.10;
const STABLE_TEXT_MIN_MARGIN: f32 = 0.05;
const STABLE_TEXT_READABLE_RATIO: f32 = 0.88;
const STABLE_TEXT_DOMINANT_RATIO: f32 = 0.60;
const STABLE_TEXT_PUNCTUATION_RATIO: f32 = 0.40;

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
    pub avg_margin: f32,
    pub min_margin: f32,
    pub char_min_confidence: f32,
    pub readable_ratio: f32,
    pub support_count: usize,
    pub source: String,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTrace {
    pub selected_source: Option<String>,
    pub det_pass_count: usize,
    pub fallback_attempt_count: usize,
    pub rec_primary_call_count: usize,
    pub rec_alt_call_count: usize,
    pub timing: OcrTraceTiming,
    pub lines: Vec<OcrTraceLine>,
    pub candidates: Vec<OcrTraceCandidate>,
    pub json: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTraceTiming {
    pub total_ms: u64,
    pub det_ms: u64,
    pub page_region_ms: u64,
    pub tile_ms: u64,
    pub color_region_ms: u64,
    pub layered_region_ms: u64,
    pub visual_region_ms: u64,
    pub fallback_ms: u64,
    pub rec_primary_ms: u64,
    pub rec_alt_ms: u64,
    pub rec_cache_hit_count: u64,
    pub rec_cache_miss_count: u64,
    pub preprocess_call_count: u64,
    pub preprocess_ms: u64,
    pub variant_candidate_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTraceLine {
    pub region_index: usize,
    pub line_index: usize,
    pub bbox: [u32; 4],
    pub crop_size: [u32; 2],
    pub text: String,
    pub confidence: f32,
    pub avg_margin: f32,
    pub min_margin: f32,
    pub char_min_confidence: f32,
    pub readable_ratio: f32,
    pub support_count: usize,
    pub source: String,
}

#[derive(Debug, Clone, Default)]
pub struct OcrTraceCandidate {
    pub label: String,
    pub mode: String,
    pub action: String,
    pub reason: String,
    pub score: f32,
    pub confidence: f32,
    pub char_count: usize,
    pub line_count: usize,
    pub region_count: usize,
    pub source_family_count: usize,
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum RecVariantForCache {
    Primary,
    Alt,
}

impl From<RecVariant> for RecVariantForCache {
    fn from(value: RecVariant) -> Self {
        match value {
            RecVariant::Primary => Self::Primary,
            RecVariant::Alt => Self::Alt,
        }
    }
}

#[derive(Debug, Clone)]
struct RecCandidate {
    text: String,
    confidence: f32,
    variant: RecVariant,
    avg_margin: f32,
    min_margin: f32,
    char_min_confidence: f32,
}

#[derive(Debug, Clone)]
struct PreparedRecognitionImage {
    image: Arc<DynamicImage>,
    signature: u64,
}

impl PreparedRecognitionImage {
    fn as_image(&self) -> &DynamicImage {
        self.image.as_ref()
    }

    fn signature(&self) -> u64 {
        self.signature
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CtcDecodeStats {
    avg_margin: f32,
    min_margin: f32,
    char_min_confidence: f32,
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

#[derive(Debug, Clone)]
struct OcrCandidateEntry {
    label: String,
    recognized: RecognizedText,
}

#[derive(Debug, Clone, Default)]
struct DetectedText {
    det_box_count: usize,
    boxes: Vec<DetectionBox>,
    recognized: RecognizedText,
}

#[derive(Debug, Clone, Default)]
struct DetectionBox {
    bbox: BoxRect,
    alternatives: Vec<BoxRect>,
}

#[derive(Debug, Clone)]
struct TextLine {
    bbox: BoxRect,
    text: String,
    confidence: f32,
    avg_margin: f32,
    min_margin: f32,
    char_min_confidence: f32,
    readable_ratio: f32,
    support_count: usize,
    source: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct OcrRecPerf {
    primary_call_count: usize,
    alt_call_count: usize,
    primary_ms: u64,
    alt_ms: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct OcrWorkPerf {
    rec_cache_hit_count: u64,
    rec_cache_miss_count: u64,
    preprocess_call_count: u64,
    preprocess_ms: u64,
    variant_candidate_count: u64,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct ImageSignatureCacheKey {
    format: u8,
    width: u32,
    height: u32,
    ptr: usize,
    len: usize,
}

#[derive(Debug, Default)]
struct RecImageSignatureCache {
    entries: HashMap<ImageSignatureCacheKey, u64>,
}

impl RecImageSignatureCache {
    fn get(&self, key: &ImageSignatureCacheKey) -> Option<u64> {
        self.entries.get(key).copied()
    }

    fn put(&mut self, key: ImageSignatureCacheKey, signature: u64) {
        if self.entries.len() >= MAX_REC_IMAGE_SIGNATURE_CACHE_ENTRIES {
            self.entries.clear();
        }
        self.entries.insert(key, signature);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct PreprocessCacheKey {
    image_signature: u64,
    target_w: usize,
    target_h: usize,
}

#[derive(Debug, Clone)]
struct CachedPreprocessResult {
    input: Arc<Vec<f32>>,
    shape: Vec<usize>,
}

#[derive(Debug, Default)]
struct PreprocessCache {
    entries: HashMap<PreprocessCacheKey, CachedPreprocessResult>,
    lru: VecDeque<PreprocessCacheKey>,
}

impl PreprocessCache {
    fn get(&mut self, key: &PreprocessCacheKey) -> Option<CachedPreprocessResult> {
        if let Some(result) = self.entries.get(key).cloned() {
            self.touch(key);
            return Some(CachedPreprocessResult {
                input: Arc::clone(&result.input),
                shape: result.shape,
            });
        }
        None
    }

    fn put(&mut self, key: PreprocessCacheKey, input: Vec<f32>, shape: Vec<usize>) {
        if self.entries.len() >= MAX_REC_PREPROCESS_CACHE_ENTRIES
            && !self.entries.contains_key(&key)
        {
            self.evict_one();
        }
        self.entries.insert(
            key,
            CachedPreprocessResult {
                input: Arc::new(input),
                shape,
            },
        );
        self.touch(&key);
    }

    fn touch(&mut self, key: &PreprocessCacheKey) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(*key);
    }

    fn evict_one(&mut self) {
        if let Some(evict) = self.lru.pop_front() {
            self.entries.remove(&evict);
        } else {
            self.entries.clear();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct ImageRgbCacheKey {
    format: u8,
    width: u32,
    height: u32,
    ptr: usize,
    len: usize,
    background: u8,
}

#[derive(Debug, Default)]
struct RgbImageCache {
    entries: HashMap<ImageRgbCacheKey, Arc<image::RgbImage>>,
    lru: VecDeque<ImageRgbCacheKey>,
}

impl RgbImageCache {
    fn get(&mut self, key: &ImageRgbCacheKey) -> Option<Arc<image::RgbImage>> {
        if let Some(image) = self.entries.get(key).cloned() {
            self.touch(key);
            return Some(image);
        }
        None
    }

    fn put(&mut self, key: ImageRgbCacheKey, image: image::RgbImage) {
        if self.entries.len() >= MAX_RGB_IMAGE_CACHE_ENTRIES && !self.entries.contains_key(&key) {
            self.evict_one();
        }
        self.entries.insert(key, Arc::new(image));
        self.touch(&key);
    }

    fn touch(&mut self, key: &ImageRgbCacheKey) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(*key);
    }

    fn evict_one(&mut self) {
        if let Some(evict) = self.lru.pop_front() {
            self.entries.remove(&evict);
        } else {
            self.entries.clear();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
struct ImageLumaCacheKey {
    format: u8,
    width: u32,
    height: u32,
    ptr: usize,
    len: usize,
    background: u8,
}

#[derive(Debug, Default)]
struct LumaImageCache {
    entries: HashMap<ImageLumaCacheKey, Arc<GrayImage>>,
    lru: VecDeque<ImageLumaCacheKey>,
}

impl LumaImageCache {
    fn get(&mut self, key: &ImageLumaCacheKey) -> Option<Arc<GrayImage>> {
        if let Some(image) = self.entries.get(key).cloned() {
            self.touch(key);
            return Some(image);
        }
        None
    }

    fn put(&mut self, key: ImageLumaCacheKey, image: GrayImage) {
        if self.entries.len() >= MAX_LUMA_IMAGE_CACHE_ENTRIES && !self.entries.contains_key(&key) {
            self.evict_one();
        }
        self.entries.insert(key, Arc::new(image));
        self.touch(&key);
    }

    fn touch(&mut self, key: &ImageLumaCacheKey) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(*key);
    }

    fn evict_one(&mut self) {
        if let Some(evict) = self.lru.pop_front() {
            self.entries.remove(&evict);
        } else {
            self.entries.clear();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

#[derive(Debug, Clone)]
struct PreparedRecognitionImageCacheEntry {
    signature: u64,
    image: Arc<DynamicImage>,
}

#[derive(Debug, Default)]
struct PreparedRecognitionImageCache {
    entries: HashMap<u64, PreparedRecognitionImageCacheEntry>,
    lru: VecDeque<u64>,
}

impl PreparedRecognitionImageCache {
    fn get(&mut self, source_signature: &u64) -> Option<PreparedRecognitionImageCacheEntry> {
        if let Some(entry) = self.entries.get(source_signature).cloned() {
            self.touch(source_signature);
            return Some(entry);
        }
        None
    }

    fn put(&mut self, source_signature: u64, entry: PreparedRecognitionImageCacheEntry) {
        if self.entries.len() >= MAX_REC_PREPARED_IMAGE_CACHE_ENTRIES
            && !self.entries.contains_key(&source_signature)
        {
            self.evict_one();
        }
        self.entries.insert(source_signature, entry);
        self.touch(&source_signature);
    }

    fn touch(&mut self, key: &u64) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(*key);
    }

    fn evict_one(&mut self) {
        if let Some(evict) = self.lru.pop_front() {
            self.entries.remove(&evict);
        } else {
            self.entries.clear();
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

#[derive(Default)]
struct OcrWorkContextScope;

thread_local! {
    static OCR_WORK_PERF: RefCell<OcrWorkPerf> = RefCell::new(OcrWorkPerf::default());
    static OCR_REC_IMAGE_SIGNATURE_CACHE: RefCell<RecImageSignatureCache> = RefCell::new(RecImageSignatureCache::default());
    static OCR_REC_PREPROCESS_CACHE: RefCell<PreprocessCache> = RefCell::new(PreprocessCache::default());
    static OCR_REC_CANDIDATE_CACHE: RefCell<RecCandidateCache> = RefCell::new(RecCandidateCache::default());
    static OCR_REC_RGB_CACHE: RefCell<RgbImageCache> = RefCell::new(RgbImageCache::default());
    static OCR_REC_LUMA_CACHE: RefCell<LumaImageCache> = RefCell::new(LumaImageCache::default());
    static OCR_REC_PREPARED_IMAGE_CACHE: RefCell<PreparedRecognitionImageCache> =
        RefCell::new(PreparedRecognitionImageCache::default());
}

impl OcrWorkContextScope {
    fn enter() -> Self {
        OCR_WORK_PERF.with(|perf| {
            *perf.borrow_mut() = OcrWorkPerf::default();
        });
        OCR_REC_IMAGE_SIGNATURE_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_PREPROCESS_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_CANDIDATE_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_RGB_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_LUMA_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_PREPARED_IMAGE_CACHE.with(|cache| cache.borrow_mut().clear());
        Self
    }
}

impl Drop for OcrWorkContextScope {
    fn drop(&mut self) {
        OCR_WORK_PERF.with(|perf| {
            *perf.borrow_mut() = OcrWorkPerf::default();
        });
        OCR_REC_IMAGE_SIGNATURE_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_PREPROCESS_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_CANDIDATE_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_RGB_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_LUMA_CACHE.with(|cache| cache.borrow_mut().clear());
        OCR_REC_PREPARED_IMAGE_CACHE.with(|cache| cache.borrow_mut().clear());
    }
}

thread_local! {
    static OCR_REC_PERF: RefCell<OcrRecPerf> = RefCell::new(OcrRecPerf::default());
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct RecCandidateCacheKey {
    image_signature: u64,
    target_w: usize,
    rec_img_h: usize,
    rec_img_w: usize,
    variant: RecVariantForCache,
}

#[derive(Debug, Default)]
struct RecCandidateCache {
    candidates: HashMap<RecCandidateCacheKey, RecCandidate>,
    lru: VecDeque<RecCandidateCacheKey>,
}

impl RecCandidateCache {
    fn get(&mut self, key: &RecCandidateCacheKey) -> Option<&RecCandidate> {
        if self.candidates.contains_key(key) {
            self.touch(key);
        }
        self.candidates.get(key)
    }

    fn put(&mut self, key: RecCandidateCacheKey, candidate: RecCandidate) {
        if !self.candidates.contains_key(&key)
            && self.candidates.len() >= MAX_REC_CANDIDATE_CACHE_ENTRIES
        {
            self.evict_one();
        }
        self.candidates.insert(key, candidate);
        self.touch(&key);
    }

    fn touch(&mut self, key: &RecCandidateCacheKey) {
        self.lru.retain(|entry| entry != key);
        self.lru.push_back(*key);
    }

    fn evict_one(&mut self) {
        if let Some(evict) = self.lru.pop_front() {
            self.candidates.remove(&evict);
        } else {
            self.candidates.clear();
        }
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.lru.clear();
    }
}

fn make_text_line(
    bbox: BoxRect,
    text: String,
    confidence: f32,
    avg_margin: f32,
    min_margin: f32,
    source: String,
) -> TextLine {
    let readable_ratio = readable_ratio(&text);
    TextLine {
        bbox,
        text,
        confidence,
        avg_margin,
        min_margin,
        char_min_confidence: confidence,
        readable_ratio,
        support_count: 1,
        source,
    }
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
    Offset {
        dx: u32,
        dy: u32,
        max_w: u32,
        max_h: u32,
    },
    ScaleOffset {
        sx: f32,
        sy: f32,
        dx: u32,
        dy: u32,
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
        let total_start = Instant::now();
        let _rec_cache_scope = OcrWorkContextScope::enter();
        reset_ocr_rec_perf();
        let trace_enabled = ocr_trace_enabled();
        let (_, img_h) = img.dimensions();
        let source_has_alpha = has_non_opaque_alpha(img);
        if trace_enabled {
            let (w, h) = img.dimensions();
            eprintln!("[OCR_TRACE] start dims={w}x{h} alpha={source_has_alpha}");
        }
        let det_start = Instant::now();
        let detected = self
            .recognize_detected_text(img, cfg, true, "det", BboxTransform::Identity)
            .map_err(|e| format!("detect: {e}"))?;
        let det_ms = elapsed_ms(det_start);
        let det_box_count = detected.det_box_count;
        let detect_used_whole_image_box = det_box_count == 0;
        let mut text = detected.recognized.text.clone();
        let mut confidence = detected.recognized.confidence;
        let mut line_count = detected.recognized.line_count;
        let mut region_count = detected.recognized.region_count;
        let mut layout_applied = detected.recognized.layout_applied;
        let mut regions = detected.recognized.regions.clone();
        let mut fallback = None;
        let mut trace = OcrTrace {
            selected_source: if text.trim().is_empty() {
                None
            } else {
                Some("det".to_string())
            },
            det_pass_count: 1,
            fallback_attempt_count: 0,
            rec_primary_call_count: 0,
            rec_alt_call_count: 0,
            timing: OcrTraceTiming {
                det_ms,
                ..OcrTraceTiming::default()
            },
            lines: Vec::new(),
            candidates: Vec::new(),
            json: None,
        };
        let mut candidate_pool = Vec::new();
        push_recognition_candidate(&mut candidate_pool, "det".to_string(), &detected.recognized);
        let mut remaining_supplement_rec_budget = MAX_EAGER_SUPPLEMENT_RECOGNITIONS_TOTAL;
        let remaining_supplement_det_budget = MAX_EAGER_SUPPLEMENT_DET_PASSES_TOTAL;
        let mut supplement_seen_boxes = SupplementSeenIndex::new(img_h);

        let page_region_start = Instant::now();
        let (page_region_count, page_region_candidate) =
            self.recognize_page_regions(img, cfg, &detected.boxes)?;
        trace.timing.page_region_ms = elapsed_ms(page_region_start);
        if page_region_count > 0 {
            trace.det_pass_count += page_region_count;
            let candidate = if text.trim().is_empty() {
                page_region_candidate
            } else {
                filter_page_region_supplement(&page_region_candidate, &regions)
            };
            maybe_adopt_recognized_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut trace,
                "det-page-regions".to_string(),
                &candidate,
            );
            maybe_adopt_candidate_pool_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut candidate_pool,
                Some(img),
                &mut trace,
                "det-page-regions".to_string(),
                &candidate,
            );
        }

        if should_use_high_res_tile_supplement(
            img,
            cfg,
            &text,
            confidence,
            det_box_count,
            line_count,
            &regions,
        ) {
            let tile_start = Instant::now();
            let (tile_region_count, tile_region_candidate) =
                self.recognize_high_res_tiles(img, cfg, &regions)?;
            trace.timing.tile_ms = elapsed_ms(tile_start);
            if tile_region_count > 0 {
                trace.det_pass_count += tile_region_count;
                maybe_adopt_recognized_traced(
                    &mut text,
                    &mut confidence,
                    &mut line_count,
                    &mut region_count,
                    &mut layout_applied,
                    &mut regions,
                    &mut fallback,
                    &mut trace,
                    "det-high-res-tiles".to_string(),
                    &tile_region_candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    &mut text,
                    &mut confidence,
                    &mut line_count,
                    &mut region_count,
                    &mut layout_applied,
                    &mut regions,
                    &mut fallback,
                    &mut candidate_pool,
                    Some(img),
                    &mut trace,
                    "det-high-res-tiles".to_string(),
                    &tile_region_candidate,
                );
            }
        }

        let mut color_region_count = 0usize;
        let run_eager_color = should_use_eager_color_region_supplement(
            &text,
            confidence,
            det_box_count,
            line_count,
            &regions,
        );
        let mut enforce_visual_progress = false;
        let mut visual_progress_anchor = (text.clone(), confidence, line_count);

        if run_eager_color
            && should_continue_eager_supplements(
                &text,
                confidence,
                det_box_count,
                line_count,
                &regions,
            )
            && remaining_supplement_rec_budget > 0
        {
            let color_before_text = text.clone();
            let color_before_confidence = confidence;
            let color_before_line_count = line_count;
            let color_region_start = Instant::now();
            let color_limit =
                MAX_EAGER_COLOR_REGION_RECOGNITIONS.min(remaining_supplement_rec_budget);
            let (candidate_count, attempted, candidate) = self.recognize_uncovered_color_regions(
                img,
                cfg,
                &regions,
                color_limit,
                "color-region:eager",
                &mut supplement_seen_boxes,
            );
            remaining_supplement_rec_budget =
                remaining_supplement_rec_budget.saturating_sub(attempted);
            maybe_adopt_recognized_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut trace,
                "color-regions:eager".to_string(),
                &candidate,
            );
            maybe_adopt_candidate_pool_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut candidate_pool,
                Some(img),
                &mut trace,
                "color-regions:eager".to_string(),
                &candidate,
            );
            color_region_count = candidate_count;
            trace.timing.color_region_ms = elapsed_ms(color_region_start);

            enforce_visual_progress = should_continue_eager_supplement_pass(
                &text,
                confidence,
                det_box_count,
                line_count,
                &regions,
                &color_before_text,
                color_before_confidence,
                color_before_line_count,
            );
            visual_progress_anchor = (
                color_before_text.clone(),
                color_before_confidence,
                color_before_line_count,
            );

            if should_continue_eager_supplement_pass(
                &text,
                confidence,
                det_box_count,
                line_count,
                &regions,
                &color_before_text,
                color_before_confidence,
                color_before_line_count,
            ) && remaining_supplement_det_budget > 0
            {
                let det_before_text = text.clone();
                let det_before_confidence = confidence;
                let det_before_line_count = line_count;
                let det_color_region_start = Instant::now();
                let det_limit =
                    MAX_EAGER_COLOR_REGION_DET_PASSES.min(remaining_supplement_det_budget);
                let (det_color_region_count, det_pass_count, candidate) = self
                    .recognize_uncovered_color_region_detections(
                        img,
                        cfg,
                        &regions,
                        det_limit,
                        "color-region-det:eager",
                        &mut supplement_seen_boxes,
                    );
                trace.timing.color_region_ms = trace
                    .timing
                    .color_region_ms
                    .saturating_add(elapsed_ms(det_color_region_start));
                color_region_count = color_region_count.max(det_color_region_count);
                if det_pass_count > 0 {
                    trace.det_pass_count += det_pass_count;
                    maybe_adopt_recognized_traced(
                        &mut text,
                        &mut confidence,
                        &mut line_count,
                        &mut region_count,
                        &mut layout_applied,
                        &mut regions,
                        &mut fallback,
                        &mut trace,
                        "color-region-det:eager".to_string(),
                        &candidate,
                    );
                    maybe_adopt_candidate_pool_traced(
                        &mut text,
                        &mut confidence,
                        &mut line_count,
                        &mut region_count,
                        &mut layout_applied,
                        &mut regions,
                        &mut fallback,
                        &mut candidate_pool,
                        Some(img),
                        &mut trace,
                        "color-region-det:eager".to_string(),
                        &candidate,
                    );
                }
                enforce_visual_progress = should_continue_eager_supplement_pass(
                    &text,
                    confidence,
                    det_box_count,
                    line_count,
                    &regions,
                    &det_before_text,
                    det_before_confidence,
                    det_before_line_count,
                );
                visual_progress_anchor = (
                    det_before_text,
                    det_before_confidence,
                    det_before_line_count,
                );
            }

            if should_continue_eager_supplement_pass(
                &text,
                confidence,
                det_box_count,
                line_count,
                &regions,
                &visual_progress_anchor.0,
                visual_progress_anchor.1,
                visual_progress_anchor.2,
            ) && remaining_supplement_rec_budget > 0
            {
                let layered_before_text = text.clone();
                let layered_before_confidence = confidence;
                let layered_before_line_count = line_count;
                let layered_region_start = Instant::now();
                let layered_limit =
                    MAX_EAGER_LAYERED_REGION_RECOGNITIONS.min(remaining_supplement_rec_budget);
                let (layered_region_count, _attempted, layered_candidate) = self
                    .recognize_layered_color_regions(
                        img,
                        cfg,
                        &regions,
                        layered_limit,
                        "layered-region:eager",
                        &mut supplement_seen_boxes,
                    );
                trace.timing.layered_region_ms = elapsed_ms(layered_region_start);
                color_region_count = color_region_count.max(layered_region_count);
                maybe_adopt_recognized_traced(
                    &mut text,
                    &mut confidence,
                    &mut line_count,
                    &mut region_count,
                    &mut layout_applied,
                    &mut regions,
                    &mut fallback,
                    &mut trace,
                    "layered-regions:eager".to_string(),
                    &layered_candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    &mut text,
                    &mut confidence,
                    &mut line_count,
                    &mut region_count,
                    &mut layout_applied,
                    &mut regions,
                    &mut fallback,
                    &mut candidate_pool,
                    Some(img),
                    &mut trace,
                    "layered-regions:eager".to_string(),
                    &layered_candidate,
                );
                enforce_visual_progress = should_continue_eager_supplement_pass(
                    &text,
                    confidence,
                    det_box_count,
                    line_count,
                    &regions,
                    &layered_before_text,
                    layered_before_confidence,
                    layered_before_line_count,
                );
                visual_progress_anchor = (
                    layered_before_text,
                    layered_before_confidence,
                    layered_before_line_count,
                );
            }
        }

        let mut can_run_visual = should_continue_eager_supplements(
            &text,
            confidence,
            det_box_count,
            line_count,
            &regions,
        ) && remaining_supplement_rec_budget > 0
            && should_use_uncovered_visual_supplement(
                img,
                cfg,
                &text,
                det_box_count,
                line_count,
                &regions,
            );
        if enforce_visual_progress {
            can_run_visual = can_run_visual
                && should_continue_eager_supplement_pass(
                    &text,
                    confidence,
                    det_box_count,
                    line_count,
                    &regions,
                    &visual_progress_anchor.0,
                    visual_progress_anchor.1,
                    visual_progress_anchor.2,
                );
        }
        if can_run_visual {
            let visual_start = Instant::now();
            let visual_limit =
                MAX_EAGER_VISUAL_REGION_RECOGNITIONS.min(remaining_supplement_rec_budget);
            let (_attempted, visual_candidate) = self.recognize_uncovered_visual_regions(
                img,
                cfg,
                &regions,
                visual_limit,
                "visual-region:eager",
                &mut supplement_seen_boxes,
            );
            trace.timing.visual_region_ms = elapsed_ms(visual_start);
            maybe_adopt_recognized_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut trace,
                "visual-regions:eager".to_string(),
                &visual_candidate,
            );
            maybe_adopt_candidate_pool_traced(
                &mut text,
                &mut confidence,
                &mut line_count,
                &mut region_count,
                &mut layout_applied,
                &mut regions,
                &mut fallback,
                &mut candidate_pool,
                Some(img),
                &mut trace,
                "visual-regions:eager".to_string(),
                &visual_candidate,
            );
        }

        if needs_quality_fallback(&text, confidence, det_box_count, line_count) {
            let fallback_start = Instant::now();
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
                &mut candidate_pool,
            )?;
            trace.timing.fallback_ms = elapsed_ms(fallback_start);
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
        let rec_perf = read_ocr_rec_perf();
        let work_perf = ocr_work_perf_snapshot();
        trace.rec_primary_call_count = rec_perf.primary_call_count;
        trace.rec_alt_call_count = rec_perf.alt_call_count;
        trace.timing.rec_primary_ms = rec_perf.primary_ms;
        trace.timing.rec_alt_ms = rec_perf.alt_ms;
        trace.timing.rec_cache_hit_count = work_perf.rec_cache_hit_count;
        trace.timing.rec_cache_miss_count = work_perf.rec_cache_miss_count;
        trace.timing.preprocess_call_count = work_perf.preprocess_call_count;
        trace.timing.preprocess_ms = work_perf.preprocess_ms;
        trace.timing.variant_candidate_count = work_perf.variant_candidate_count;
        trace.timing.total_ms = elapsed_ms(total_start);
        trace.lines = ocr_trace_lines_from_regions(&regions);
        if ocr_trace_json_enabled() {
            let (w, h) = img.dimensions();
            let json = ocr_trace_json(
                w,
                h,
                source_has_alpha,
                det_box_count,
                color_region_count,
                detect_used_whole_image_box,
                empty_result,
                confidence,
                &trace,
                &regions,
            );
            eprintln!("[OCR_TRACE_JSON] {json}");
            trace.json = Some(json);
        }

        if trace_enabled {
            let (w, h) = img.dimensions();
            eprintln!(
                "[OCR_TRACE] dims={}x{} alpha={} det_boxes={} line_count={} regions={} layout={} color_regions={} det_passes={} fallback_attempts={} rec_primary_calls={} rec_alt_calls={} rec_cache_hits={} rec_cache_miss={} preprocess_calls={} preprocess_ms={} variant_candidates={} source={} whole_image_box={} fallback={} empty={} total_ms={} det_ms={} page_region_ms={} tile_ms={} color_region_ms={} layered_region_ms={} visual_region_ms={} fallback_ms={} rec_primary_ms={} rec_alt_ms={}",
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
                trace.rec_primary_call_count,
                trace.rec_alt_call_count,
                trace.timing.rec_cache_hit_count,
                trace.timing.rec_cache_miss_count,
                trace.timing.preprocess_call_count,
                trace.timing.preprocess_ms,
                trace.timing.variant_candidate_count,
                trace.selected_source.as_deref().unwrap_or("-"),
                detect_used_whole_image_box,
                fallback.as_deref().unwrap_or("-"),
                empty_result,
                trace.timing.total_ms,
                trace.timing.det_ms,
                trace.timing.page_region_ms,
                trace.timing.tile_ms,
                trace.timing.color_region_ms,
                trace.timing.layered_region_ms,
                trace.timing.visual_region_ms,
                trace.timing.fallback_ms,
                trace.timing.rec_primary_ms,
                trace.timing.rec_alt_ms
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
        let include_raw_split_candidates =
            source == "det" || source.starts_with("color-region-det:");
        let boxes = self.detect_text_boxes(img, cfg, include_raw_split_candidates)?;
        let trace_enabled = ocr_trace_enabled();
        if trace_enabled {
            eprintln!(
                "[OCR_TRACE] det-pass source={} boxes={} crop_enhance_budget={} split_budget={} repair_budget={}",
                source,
                boxes.len(),
                if allow_crop_enhancement {
                    MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS
                } else {
                    0
                },
                split_line_recognition_budget(source),
                line_repair_recognition_budget(source)
            );
        }
        let mut lines = Vec::new();
        let mut crop_enhancement_budget = if allow_crop_enhancement {
            MAX_CROP_ENHANCEMENT_ATTEMPTS_PER_PASS
        } else {
            0
        };
        let mut split_line_rec_budget = split_line_recognition_budget(source);
        let mut line_repair_rec_budget = line_repair_recognition_budget(source);
        for (idx, det_box) in boxes.iter().enumerate() {
            let b = det_box.bbox;
            if trace_enabled {
                eprintln!(
                    "[OCR_TRACE] det-pass-box source={} index={} bbox={}x{}@{},{}",
                    source,
                    idx + 1,
                    box_width(b),
                    box_height(b),
                    b.0,
                    b.1
                );
            }
            self.push_recognized_box_lines(
                img,
                cfg,
                b,
                &det_box.alternatives,
                allow_crop_enhancement,
                source,
                transform,
                &mut crop_enhancement_budget,
                &mut split_line_rec_budget,
                &mut line_repair_rec_budget,
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

        let recognized = if matches!(transform, BboxTransform::Identity) {
            recognized_from_text_lines_with_context(&mut lines, Some(img))
        } else {
            recognized_from_text_lines(&mut lines)
        };

        Ok(DetectedText {
            det_box_count: boxes.len(),
            boxes,
            recognized,
        })
    }

    fn recognize_page_regions(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        base_boxes: &[DetectionBox],
    ) -> Result<(usize, RecognizedText), String> {
        let region_boxes = page_region_boxes(img, base_boxes);
        let trace_enabled = ocr_trace_enabled();
        if trace_enabled {
            eprintln!("[OCR_TRACE] page-regions candidates={}", region_boxes.len());
        }
        if region_boxes.is_empty() {
            return Ok((0, RecognizedText::default()));
        }

        let (img_w, img_h) = img.dimensions();
        let mut lines = Vec::new();
        let mut det_passes = 0usize;
        for (idx, region_box) in region_boxes.iter().enumerate() {
            if trace_enabled {
                eprintln!(
                    "[OCR_TRACE] page-region index={} bbox={}x{}@{},{}",
                    idx + 1,
                    box_width(*region_box),
                    box_height(*region_box),
                    region_box.0,
                    region_box.1
                );
            }
            let source = format!("page-region:{}", idx + 1);
            let (region_passes, recognized) = self.recognize_panel_region_recursive(
                img,
                cfg,
                *region_box,
                0,
                &source,
                (region_box.0, region_box.1),
                (img_w, img_h),
            )?;
            det_passes += region_passes;
            lines.extend(text_lines_from_recognized(&recognized));
        }

        Ok((det_passes, recognized_from_text_lines(&mut lines)))
    }

    fn recognize_panel_region_recursive(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        region_box: BoxRect,
        depth: usize,
        source: &str,
        origin: (u32, u32),
        image_dims: (u32, u32),
    ) -> Result<(usize, RecognizedText), String> {
        let crop = crop_box(img, region_box);
        let mut det_passes = 1usize;
        let transform = BboxTransform::Offset {
            dx: origin.0,
            dy: origin.1,
            max_w: image_dims.0,
            max_h: image_dims.1,
        };
        let detected = self.recognize_detected_text(&crop, cfg, false, source, transform)?;
        let mut recognized = detected.recognized.clone();

        if depth + 1 < MAX_PANEL_RECURSION_DEPTH {
            let child_boxes = panel_child_candidate_boxes(&crop);
            if child_boxes.len() >= 2 {
                let mut child_lines = Vec::new();
                for (idx, child_box) in child_boxes.iter().enumerate() {
                    let child_source = format!("{source}.{}", idx + 1);
                    let (child_passes, child_recognized) = self.recognize_panel_region_recursive(
                        &crop,
                        cfg,
                        *child_box,
                        depth + 1,
                        &child_source,
                        (
                            origin.0.saturating_add(child_box.0),
                            origin.1.saturating_add(child_box.1),
                        ),
                        image_dims,
                    )?;
                    det_passes += child_passes;
                    child_lines.extend(text_lines_from_recognized(&child_recognized));
                }
                if !child_lines.is_empty() {
                    let child_candidate = recognized_from_text_lines(&mut child_lines);
                    recognized = merge_recognized_line_sets(&recognized.regions, &child_candidate);
                }
            }
        }

        if should_try_low_threshold_panel_det(&crop, &recognized) {
            let mut low_cfg = cfg.clone();
            low_cfg.det_box_thresh = low_threshold_box_thresh(cfg.det_box_thresh);
            let low_source = format!("{source}:low-det");
            let low_detected =
                self.recognize_detected_text(&crop, &low_cfg, false, &low_source, transform)?;
            det_passes += 1;
            if !low_detected.recognized.text.trim().is_empty() {
                recognized =
                    merge_recognized_line_sets(&recognized.regions, &low_detected.recognized);
            }
        }

        Ok((det_passes, recognized))
    }

    fn recognize_high_res_tiles(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        existing_regions: &[OcrTextRegion],
    ) -> Result<(usize, RecognizedText), String> {
        let tile_boxes = high_res_tile_boxes(img, cfg.det_img_side);
        let trace_enabled = ocr_trace_enabled();
        if trace_enabled {
            eprintln!("[OCR_TRACE] high-res-tiles candidates={}", tile_boxes.len());
        }
        if tile_boxes.is_empty() {
            return Ok((0, RecognizedText::default()));
        }

        let (img_w, img_h) = img.dimensions();
        let mut lines = Vec::new();
        for (idx, tile_box) in tile_boxes.iter().enumerate() {
            if trace_enabled {
                eprintln!(
                    "[OCR_TRACE] high-res-tile index={} bbox={}x{}@{},{}",
                    idx + 1,
                    box_width(*tile_box),
                    box_height(*tile_box),
                    tile_box.0,
                    tile_box.1
                );
            }
            let crop = crop_box(img, *tile_box);
            let source = format!("tile-region:{}", idx + 1);
            let detected = self.recognize_detected_text(
                &crop,
                cfg,
                false,
                &source,
                BboxTransform::Offset {
                    dx: tile_box.0,
                    dy: tile_box.1,
                    max_w: img_w,
                    max_h: img_h,
                },
            )?;
            lines.extend(text_lines_from_recognized(&detected.recognized));
        }

        let recognized = recognized_from_text_lines(&mut lines);
        let candidate = if existing_regions.is_empty() {
            recognized
        } else {
            filter_page_region_supplement(&recognized, existing_regions)
        };
        Ok((tile_boxes.len(), candidate))
    }

    fn push_recognized_box_lines(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        b: BoxRect,
        alternative_boxes: &[BoxRect],
        allow_crop_enhancement: bool,
        source: &str,
        transform: BboxTransform,
        crop_enhancement_budget: &mut usize,
        split_line_rec_budget: &mut usize,
        line_repair_rec_budget: &mut usize,
        lines: &mut Vec<TextLine>,
    ) {
        let using_alternatives =
            alternative_boxes.len() >= 2 && alternative_boxes.len() <= *split_line_rec_budget;
        let mut split_boxes = if using_alternatives {
            alternative_boxes.to_vec()
        } else {
            split_text_box_into_color_region_boxes(img, b)
        };
        if !using_alternatives
            && (split_boxes.len() < 2 || split_boxes.len() > *split_line_rec_budget)
        {
            split_boxes = split_text_box_into_line_boxes(img, b);
        }
        split_boxes = dedupe_box_candidates(split_boxes);
        if split_boxes.len() >= 2 && split_boxes.len() <= *split_line_rec_budget {
            let split_lines =
                self.recognize_split_line_boxes(img, cfg, &split_boxes, source, transform);
            let direct_for_comparison = if using_alternatives {
                let direct_crop = crop_box(img, b);
                self.best_from_crop_direct(&direct_crop, cfg)
            } else {
                None
            };
            if should_use_split_lines(direct_for_comparison.as_ref(), &split_lines) {
                *split_line_rec_budget -= split_boxes.len();
                lines.extend(split_lines);
                return;
            }
            if let Some(candidate) = direct_for_comparison {
                lines.extend(candidate_text_lines(img, b, &candidate, source, transform));
                return;
            }
        }

        let forced_budget = (*split_line_rec_budget)
            .saturating_add(*line_repair_rec_budget)
            .min(MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS + MAX_LINE_REPAIR_RECOGNITIONS_PER_PASS);
        if large_text_box_should_prioritize_split(b) && forced_budget >= 2 {
            let forced_boxes = forced_structural_split_boxes(img, b, forced_budget);
            if forced_boxes.len() >= 2 {
                let forced_source = format!("{source}:forced");
                let split_lines = self.recognize_split_line_boxes(
                    img,
                    cfg,
                    &forced_boxes,
                    &forced_source,
                    transform,
                );
                if structured_split_lines_are_plausible(b, &split_lines) {
                    consume_recognition_budget(
                        forced_boxes.len(),
                        split_line_rec_budget,
                        line_repair_rec_budget,
                    );
                    lines.extend(split_lines);
                    return;
                }
            }
        }

        let direct_crop = crop_box(img, b);
        let mut direct = self.best_from_crop_direct(&direct_crop, cfg);
        if large_text_box_needs_structured_split(b) {
            let forced_boxes = forced_structural_split_boxes(img, b, forced_budget);
            if forced_boxes.len() >= 2 {
                let forced_source = format!("{source}:forced");
                let split_lines = self.recognize_split_line_boxes(
                    img,
                    cfg,
                    &forced_boxes,
                    &forced_source,
                    transform,
                );
                if should_use_forced_split_lines(b, direct.as_ref(), &split_lines) {
                    consume_recognition_budget(
                        forced_boxes.len(),
                        split_line_rec_budget,
                        line_repair_rec_budget,
                    );
                    lines.extend(split_lines);
                    return;
                }
            }
        }
        let direct_is_strong = direct
            .as_ref()
            .is_some_and(|candidate| candidate.confidence >= MIN_STRONG_REC_CONFIDENCE);
        if allow_crop_enhancement
            && !direct_is_strong
            && *crop_enhancement_budget > 0
            && should_enhance_crop(b)
        {
            *crop_enhancement_budget -= 1;
            direct = self
                .best_from_crop(&direct_crop, cfg, direct.clone())
                .or(direct);
        };

        if let Some(repaired) = self.repair_recognized_box_lines(
            img,
            cfg,
            b,
            direct.as_ref(),
            source,
            transform,
            line_repair_rec_budget,
        ) {
            lines.extend(repaired);
            return;
        }

        if let Some(candidate) = direct {
            lines.extend(candidate_text_lines(img, b, &candidate, source, transform));
        }
    }

    fn repair_recognized_box_lines(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        b: BoxRect,
        direct: Option<&RecCandidate>,
        source: &str,
        transform: BboxTransform,
        line_repair_rec_budget: &mut usize,
    ) -> Option<Vec<TextLine>> {
        if *line_repair_rec_budget == 0 {
            return None;
        }
        if let Some(candidate) = direct {
            let text = normalize_recognized_text(&candidate.text);
            if !recognized_box_needs_repair(b, &text, candidate.confidence) {
                return None;
            }
        }

        let split_boxes = repair_split_boxes(img, b, *line_repair_rec_budget);
        if split_boxes.len() >= 2 {
            let split_source = format!("{source}:repair");
            let split_lines =
                self.recognize_split_line_boxes(img, cfg, &split_boxes, &split_source, transform);
            if should_use_split_lines(direct, &split_lines) {
                *line_repair_rec_budget =
                    (*line_repair_rec_budget).saturating_sub(split_boxes.len());
                return Some(split_lines);
            }
        }

        if let Some(wide_lines) = self.repair_wide_line_segments(
            img,
            cfg,
            b,
            direct,
            source,
            transform,
            line_repair_rec_budget,
        ) {
            return Some(wide_lines);
        }

        let crop = crop_box(img, b);
        if let Some(candidate) = self.best_from_crop_local_preprocessed(&crop, cfg)
            && repair_candidate_is_better(direct, &candidate)
        {
            *line_repair_rec_budget = (*line_repair_rec_budget).saturating_sub(1);
            let source = format!("{source}:local");
            return Some(candidate_text_lines(img, b, &candidate, &source, transform));
        }

        None
    }

    fn repair_wide_line_segments(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        b: BoxRect,
        direct: Option<&RecCandidate>,
        source: &str,
        transform: BboxTransform,
        line_repair_rec_budget: &mut usize,
    ) -> Option<Vec<TextLine>> {
        if *line_repair_rec_budget < 2 {
            return None;
        }
        let segment_limit = wide_line_segment_limit(b, *line_repair_rec_budget);
        let segment_boxes = wide_line_recognition_boxes(img, b, segment_limit);
        if segment_boxes.len() < 2 {
            return None;
        }

        let mut segment_candidates = Vec::new();
        for segment_box in &segment_boxes {
            let crop = crop_box(img, *segment_box);
            let candidate = self.best_from_crop_local_preprocessed(&crop, cfg);
            if let Some(candidate) = candidate {
                segment_candidates.push(candidate);
            }
        }
        if segment_candidates.len() != segment_boxes.len() {
            return None;
        }

        let combined_text = join_segment_recognition_text(&segment_candidates);
        let confidence = segment_candidates
            .iter()
            .map(|candidate| candidate.confidence)
            .sum::<f32>()
            / segment_candidates.len() as f32;
        let avg_margin = segment_candidates
            .iter()
            .map(|candidate| candidate.avg_margin)
            .sum::<f32>()
            / segment_candidates.len() as f32;
        let min_margin = segment_candidates
            .iter()
            .map(|candidate| candidate.min_margin)
            .fold(f32::INFINITY, f32::min);
        let candidate = RecCandidate {
            text: combined_text,
            confidence,
            variant: segment_candidates[0].variant,
            avg_margin,
            min_margin: if min_margin.is_finite() {
                min_margin
            } else {
                0.0
            },
            char_min_confidence: segment_candidates
                .iter()
                .map(|candidate| candidate.char_min_confidence)
                .fold(f32::INFINITY, f32::min)
                .min(confidence),
        };
        if !is_usable_recognition(&candidate) || !repair_candidate_is_better(direct, &candidate) {
            return None;
        }

        *line_repair_rec_budget = (*line_repair_rec_budget).saturating_sub(segment_boxes.len());
        let source = format!("{source}:wide");
        Some(candidate_text_lines(img, b, &candidate, &source, transform))
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
            let candidate = self.best_from_crop_local_preprocessed(&crop, cfg);
            if let Some(candidate) = candidate {
                lines.extend(candidate_text_lines(
                    img,
                    *split_box,
                    &candidate,
                    &format!("{source}:split"),
                    transform,
                ));
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
        candidate_pool: &mut Vec<OcrCandidateEntry>,
    ) -> Result<(), String> {
        let image_bbox = image_box(img);
        let mut family_budget = quality_fallback_family_budget(text);
        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
        }
        trace.fallback_attempt_count += 1;
        match self.recognize_best(img, cfg) {
            Ok(candidate) if is_usable_recognition(&candidate) => {
                let label = recognition_fallback_label("whole-image", candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
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

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
        }
        trace.fallback_attempt_count += 1;
        let (candidate_count, candidate) = self.recognize_color_regions(img, cfg);
        *color_region_count = (*color_region_count).max(candidate_count);
        maybe_adopt_recognized_traced(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            trace,
            "color-regions".to_string(),
            &candidate,
        );
        maybe_adopt_candidate_pool_traced(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            candidate_pool,
            Some(img),
            trace,
            "color-regions".to_string(),
            &candidate,
        );
        if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
            return Ok(());
        }

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
        }
        let enhancement_limit = quality_fallback_enhancement_variant_budget(
            text,
            *confidence,
            det_box_count,
            *line_count,
        );
        for (name, enhanced) in enhancement_variants_limited(img, enhancement_limit) {
            trace.fallback_attempt_count += 1;
            trace.det_pass_count += 1;
            if let Ok(candidate) = self.recognize_detected_text(
                &enhanced,
                cfg,
                false,
                &format!("det-enhanced:{name}"),
                BboxTransform::Identity,
            ) {
                let label = format!("det-enhanced:{name}");
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate.recognized,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
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
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
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
                let label = format!("det-upscaled:{name}");
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate.recognized,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
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
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
        }
        for (name, deskewed) in deskew_variants(img) {
            trace.fallback_attempt_count += 1;
            if let Ok(candidate) = self.recognize_best(&deskewed, cfg)
                && is_usable_recognition(&candidate)
            {
                let label =
                    recognition_fallback_label(&format!("deskew:{name}"), candidate.variant);
                let candidate = recognized_from_candidate(candidate, image_bbox, &label);
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
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
                let label = format!("det-rotated:{name}");
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate.recognized,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
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
                maybe_adopt_recognized_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    trace,
                    label.clone(),
                    &candidate,
                );
                maybe_adopt_candidate_pool_traced(
                    text,
                    confidence,
                    line_count,
                    region_count,
                    layout_applied,
                    regions,
                    fallback,
                    candidate_pool,
                    Some(img),
                    trace,
                    label,
                    &candidate,
                );
            }
            if !needs_quality_fallback(text, *confidence, det_box_count, *line_count) {
                return Ok(());
            }
        }

        if !consume_quality_fallback_family(&mut family_budget) {
            return Ok(());
        }
        trace.fallback_attempt_count += 1;
        let candidate = self.recognize_line_crops(img, cfg);
        maybe_adopt_recognized_traced(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            trace,
            "line-crops".to_string(),
            &candidate,
        );
        maybe_adopt_candidate_pool_traced(
            text,
            confidence,
            line_count,
            region_count,
            layout_applied,
            regions,
            fallback,
            candidate_pool,
            Some(img),
            trace,
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
        let raw_boxes = color_region_boxes(img);
        let boxes = prioritize_supplement_candidate_boxes(img, raw_boxes.clone(), &[]);
        let mut lines = Vec::new();
        for b in boxes.iter().take(recognition_limit) {
            let crop = crop_box(img, *b);
            if let Some(candidate) = self.best_from_crop_local_preprocessed(&crop, cfg) {
                lines.push(make_text_line(
                    *b,
                    normalize_recognized_text(&candidate.text),
                    candidate.confidence,
                    candidate.avg_margin,
                    candidate.min_margin,
                    source.to_string(),
                ));
            }
        }
        (raw_boxes.len(), recognized_from_text_lines(&mut lines))
    }

    fn recognize_uncovered_color_regions(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        existing_regions: &[OcrTextRegion],
        recognition_limit: usize,
        source: &str,
        supplement_seen_boxes: &mut SupplementSeenIndex,
    ) -> (usize, usize, RecognizedText) {
        let raw_boxes = color_region_boxes(img);
        let boxes = prioritize_supplement_candidate_boxes(img, raw_boxes.clone(), existing_regions);
        let mut lines = Vec::new();
        let mut attempted = 0usize;
        for b in &boxes {
            if attempted >= recognition_limit {
                break;
            }
            if color_region_box_covered_by_reliable_text(*b, existing_regions) {
                continue;
            }
            if !supplement_box_is_worth_recognition(*b) {
                continue;
            }
            if supplement_seen_boxes.is_redundant(*b) {
                continue;
            }
            let mut found_reliable = false;
            let crop = crop_box(img, *b);
            attempted += 1;
            if let Some(candidate) = self.best_from_crop_local_preprocessed(&crop, cfg) {
                let candidate_lines =
                    candidate_text_lines(img, *b, &candidate, source, BboxTransform::Identity);
                if supplement_lines_are_reliable(&candidate_lines) {
                    found_reliable = true;
                }
                lines.extend(candidate_lines);
            }
            supplement_seen_boxes.insert_if_reliable(*b, found_reliable);
        }
        (
            raw_boxes.len(),
            attempted,
            recognized_from_text_lines(&mut lines),
        )
    }

    fn recognize_uncovered_color_region_detections(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        existing_regions: &[OcrTextRegion],
        recognition_limit: usize,
        source: &str,
        supplement_seen_boxes: &mut SupplementSeenIndex,
    ) -> (usize, usize, RecognizedText) {
        let (candidate_count, boxes) =
            color_region_det_candidate_boxes(img, existing_regions, recognition_limit);
        if boxes.is_empty() {
            return (candidate_count, 0, RecognizedText::default());
        }

        let (img_w, img_h) = img.dimensions();
        let mut lines = Vec::new();
        let mut det_pass_count = 0usize;
        for (idx, b) in boxes.iter().enumerate() {
            if supplement_seen_boxes.is_redundant(*b) {
                continue;
            }
            let mut found_reliable = false;
            let crop = crop_box(img, *b);
            let source = format!("{source}:{}", idx + 1);
            det_pass_count += 1;
            if let Ok(detected) = self.recognize_detected_text(
                &crop,
                cfg,
                false,
                &source,
                BboxTransform::Offset {
                    dx: b.0,
                    dy: b.1,
                    max_w: img_w,
                    max_h: img_h,
                },
            ) {
                let supplement =
                    filter_color_region_det_supplement(&detected.recognized, existing_regions);
                let supplement_lines = text_lines_from_recognized(&supplement);
                if supplement_lines_are_reliable(&supplement_lines) {
                    found_reliable = true;
                }
                lines.extend(supplement_lines);
            }
            for (name, upscaled) in local_det_upscale_variants(&crop)
                .into_iter()
                .take(MAX_LOCAL_DET_UPSCALE_PASSES_PER_REGION)
            {
                let (up_w, up_h) = upscaled.dimensions();
                if up_w == 0 || up_h == 0 {
                    continue;
                }
                det_pass_count += 1;
                if let Ok(detected) = self.recognize_detected_text(
                    &upscaled,
                    cfg,
                    false,
                    &format!("{source}:{name}"),
                    BboxTransform::ScaleOffset {
                        sx: box_width(*b) as f32 / up_w as f32,
                        sy: box_height(*b) as f32 / up_h as f32,
                        dx: b.0,
                        dy: b.1,
                        max_w: img_w,
                        max_h: img_h,
                    },
                ) {
                    let supplement =
                        filter_color_region_det_supplement(&detected.recognized, existing_regions);
                    let supplement_lines = text_lines_from_recognized(&supplement);
                    if supplement_lines_are_reliable(&supplement_lines) {
                        found_reliable = true;
                    }
                    lines.extend(supplement_lines);
                }
            }
            supplement_seen_boxes.insert_if_reliable(*b, found_reliable);
        }

        (
            candidate_count,
            det_pass_count,
            recognized_from_text_lines(&mut lines),
        )
    }

    fn recognize_layered_color_regions(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        existing_regions: &[OcrTextRegion],
        recognition_limit: usize,
        source: &str,
        supplement_seen_boxes: &mut SupplementSeenIndex,
    ) -> (usize, usize, RecognizedText) {
        let boxes = layered_color_region_text_boxes(img, existing_regions);
        let candidate_count = boxes.len();
        let mut lines = Vec::new();
        let mut attempted = 0usize;
        for b in &boxes {
            if attempted >= recognition_limit {
                break;
            }
            if color_region_box_covered_by_reliable_text(*b, existing_regions) {
                continue;
            }
            if !supplement_box_is_worth_recognition(*b) {
                continue;
            }
            if supplement_seen_boxes.is_redundant(*b) {
                continue;
            }
            let mut found_reliable = false;
            let crop = crop_box(img, *b);
            attempted += 1;
            if let Some(candidate) = self.best_from_crop_local_preprocessed(&crop, cfg) {
                let candidate_lines =
                    candidate_text_lines(img, *b, &candidate, source, BboxTransform::Identity);
                if supplement_lines_are_reliable(&candidate_lines) {
                    found_reliable = true;
                }
                lines.extend(candidate_lines);
            }
            supplement_seen_boxes.insert_if_reliable(*b, found_reliable);
        }
        (
            candidate_count,
            attempted,
            recognized_from_text_lines(&mut lines),
        )
    }

    fn recognize_uncovered_visual_regions(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        existing_regions: &[OcrTextRegion],
        recognition_limit: usize,
        source: &str,
        supplement_seen_boxes: &mut SupplementSeenIndex,
    ) -> (usize, RecognizedText) {
        let boxes = prioritize_supplement_candidate_boxes(
            img,
            uncovered_visual_text_boxes(img, existing_regions),
            existing_regions,
        );
        let mut lines = Vec::new();
        let mut attempted = 0usize;
        for b in &boxes {
            if attempted >= recognition_limit {
                break;
            }
            if !supplement_box_is_worth_recognition(*b) {
                continue;
            }
            if supplement_seen_boxes.is_redundant(*b) {
                continue;
            }
            let mut found_reliable = false;
            let crop = crop_box(img, *b);
            let candidate = self.best_from_crop_local_preprocessed(&crop, cfg);
            attempted += 1;
            if let Some(candidate) = candidate {
                let candidate_lines =
                    candidate_text_lines(img, *b, &candidate, source, BboxTransform::Identity);
                if supplement_lines_are_reliable(&candidate_lines) {
                    found_reliable = true;
                }
                lines.extend(candidate_lines);
            }
            supplement_seen_boxes.insert_if_reliable(*b, found_reliable);
        }
        (attempted, recognized_from_text_lines(&mut lines))
    }

    fn recognize_line_crops(&self, img: &DynamicImage, cfg: &OcrConfig) -> RecognizedText {
        let mut lines = Vec::new();
        for line_box in fallback_line_boxes(img) {
            let line = crop_box(img, line_box);
            if let Ok(candidate) = self.recognize_best(&line, cfg)
                && is_usable_recognition(&candidate)
            {
                lines.push(make_text_line(
                    line_box,
                    normalize_recognized_text(&candidate.text),
                    candidate.confidence,
                    candidate.avg_margin,
                    candidate.min_margin,
                    "line-crops".to_string(),
                ));
            }
        }
        recognized_from_text_lines(&mut lines)
    }

    fn recognize_best(
        &self,
        image: &DynamicImage,
        cfg: &OcrConfig,
    ) -> Result<RecCandidate, String> {
        OCR_REC_CANDIDATE_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            self.recognize_best_cached(image, cfg, &mut cache)
        })
    }

    fn recognize_best_cached(
        &self,
        image: &DynamicImage,
        cfg: &OcrConfig,
        rec_cache: &mut RecCandidateCache,
    ) -> Result<RecCandidate, String> {
        let prepared = self.prepare_recognition_input(image);
        let primary = self.recognize_candidate_prepared(
            &self.rec,
            &self.alphabet,
            &prepared,
            cfg,
            RecVariant::Primary,
            rec_cache,
        )?;
        let alt = if should_try_alt_recognition(&primary) {
            if let Some(rec_alt) = &self.rec_alt {
                Some(self.recognize_candidate_prepared(
                    rec_alt,
                    &self.alphabet_alt,
                    &prepared,
                    cfg,
                    RecVariant::Alt,
                    rec_cache,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(select_recognition(primary, alt))
    }

    fn prepare_recognition_input(&self, image: &DynamicImage) -> PreparedRecognitionImage {
        let source_signature = dynamic_image_signature_cached(image);
        if let Some(cached) =
            OCR_REC_PREPARED_IMAGE_CACHE.with(|cache| cache.borrow_mut().get(&source_signature))
        {
            return PreparedRecognitionImage {
                image: cached.image,
                signature: cached.signature,
            };
        }

        let image = if let Some(cropped) = tight_rec_crop(image) {
            let signature = dynamic_image_signature_cached(&cropped);
            PreparedRecognitionImage {
                image: Arc::new(cropped),
                signature,
            }
        } else {
            PreparedRecognitionImage {
                image: Arc::new(image.clone()),
                signature: source_signature,
            }
        };

        OCR_REC_PREPARED_IMAGE_CACHE.with(|cache| {
            cache.borrow_mut().put(
                source_signature,
                PreparedRecognitionImageCacheEntry {
                    signature: image.signature,
                    image: Arc::clone(&image.image),
                },
            );
        });
        image
    }

    fn best_from_crop(
        &self,
        image: &DynamicImage,
        cfg: &OcrConfig,
        direct: Option<RecCandidate>,
    ) -> Option<RecCandidate> {
        let mut best = direct.filter(is_usable_recognition);
        if !should_try_crop_enhancement_variants(best.as_ref()) {
            return best;
        }

        let variant_budget = crop_enhancement_variant_budget(image, best.as_ref());
        let variant_budget = variant_budget.max(1);
        let mut stale_streak = 0usize;
        for (_name, enhanced) in enhancement_variants_limited(image, variant_budget).into_iter() {
            ocr_work_perf_record_variant_candidates(1);
            if let Ok(candidate) = self.recognize_best(&enhanced, cfg) {
                let is_improved = is_usable_recognition(&candidate)
                    && recognition_candidate_is_better(best.as_ref(), &candidate);
                if !is_improved {
                    stale_streak = stale_streak.saturating_add(1);
                    if should_short_circuit_crop_enhancement(
                        best.as_ref(),
                        stale_streak,
                        variant_budget,
                    ) {
                        break;
                    }
                    continue;
                }
                best = Some(candidate);
                stale_streak = 0;
                if let Some(best_ref) = best.as_ref()
                    && crop_enhancement_candidate_is_final(best_ref)
                {
                    break;
                }

                if should_short_circuit_crop_enhancement(
                    best.as_ref(),
                    stale_streak,
                    variant_budget,
                ) {
                    break;
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

    fn best_from_crop_local_preprocessed(
        &self,
        image: &DynamicImage,
        cfg: &OcrConfig,
    ) -> Option<RecCandidate> {
        let mut best = self.best_from_crop_direct(image, cfg);
        if !should_try_local_recognition_variants(image, best.as_ref()) {
            return best;
        }
        let mut stale_streak = 0usize;
        let variant_budget = if best.is_some() { 3 } else { 4 };
        for (_, variant) in local_recognition_variants_adaptive(image, best.as_ref()) {
            ocr_work_perf_record_variant_candidates(1);
            if let Ok(candidate) = self.recognize_best(&variant, cfg)
                && is_usable_recognition(&candidate)
                && recognition_candidate_is_better(best.as_ref(), &candidate)
            {
                best = Some(candidate);
                stale_streak = 0;
                if best
                    .as_ref()
                    .is_some_and(local_recognition_candidate_is_final)
                {
                    break;
                }
            } else {
                stale_streak = stale_streak.saturating_add(1);
                if stale_streak >= 2
                    && should_short_circuit_crop_enhancement(
                        best.as_ref(),
                        stale_streak,
                        variant_budget,
                    )
                {
                    break;
                }
            }
        }
        best
    }

    fn recognize_candidate(
        &self,
        session: &OrtSession,
        alphabet: &[String],
        image: &DynamicImage,
        cfg: &OcrConfig,
        variant: RecVariant,
        rec_cache: &mut RecCandidateCache,
    ) -> Result<RecCandidate, String> {
        self.recognize_candidate_prepared(
            session,
            alphabet,
            &self.prepare_recognition_input(image),
            cfg,
            variant,
            rec_cache,
        )
    }

    fn recognize_candidate_prepared(
        &self,
        session: &OrtSession,
        alphabet: &[String],
        prepared: &PreparedRecognitionImage,
        cfg: &OcrConfig,
        variant: RecVariant,
        rec_cache: &mut RecCandidateCache,
    ) -> Result<RecCandidate, String> {
        let image = prepared.as_image();
        let image_signature = prepared.signature();
        let target_w = dynamic_rec_target_width(image, cfg.rec_img_h, cfg.rec_img_w);
        let key = rec_candidate_cache_key_with_signature(
            image_signature,
            cfg.rec_img_h,
            cfg.rec_img_w,
            target_w,
            variant,
        );
        if let Some(candidate) = rec_cache.get(&key) {
            ocr_work_perf_record_rec_cache_hit();
            return Ok(candidate.clone());
        }
        ocr_work_perf_record_rec_cache_miss();

        match self.recognize_candidate_at_width_cached(
            session,
            alphabet,
            image,
            image_signature,
            cfg,
            variant,
            target_w,
        ) {
            Ok(candidate) => {
                rec_cache.put(key, candidate.clone());
                Ok(candidate)
            }
            Err(e) if target_w != cfg.rec_img_w => {
                let fixed_key = rec_candidate_cache_key_with_signature(
                    image_signature,
                    cfg.rec_img_h,
                    cfg.rec_img_w,
                    cfg.rec_img_w,
                    variant,
                );
                if let Some(candidate) = rec_cache.get(&fixed_key).cloned() {
                    ocr_work_perf_record_rec_cache_hit();
                    rec_cache.put(key, candidate.clone());
                    return Ok(candidate);
                }
                ocr_work_perf_record_rec_cache_miss();
                let candidate = self
                    .recognize_candidate_at_width_cached(
                        session,
                        alphabet,
                        image,
                        image_signature,
                        cfg,
                        variant,
                        cfg.rec_img_w,
                    )
                    .map_err(|fallback| format!("{e}; fixed-width fallback failed: {fallback}"))?;
                rec_cache.put(fixed_key, candidate.clone());
                rec_cache.put(key, candidate.clone());
                Ok(candidate)
            }
            Err(e) => Err(e),
        }
    }

    fn recognize_candidate_at_width_cached(
        &self,
        session: &OrtSession,
        alphabet: &[String],
        image: &DynamicImage,
        image_signature: u64,
        cfg: &OcrConfig,
        variant: RecVariant,
        target_w: usize,
    ) -> Result<RecCandidate, String> {
        let start = Instant::now();
        let (rec_input, rec_shape) = preprocess_rec_image_cached_with_signature(
            image,
            image_signature,
            cfg.rec_img_h,
            target_w,
        )?;
        let input_slices = [rec_input.as_slice()];
        let shape_slices = [rec_shape.as_slice()];
        let (output, out_shapes) = ort::run_session(session, &input_slices, &shape_slices)?;
        let logits = &output[0];
        let (text, confidence, stats) = ctc_decode_with_stats(logits, &out_shapes[0], alphabet);
        record_ocr_rec_perf(variant, elapsed_ms(start));
        Ok(RecCandidate {
            text,
            confidence,
            variant,
            avg_margin: stats.avg_margin,
            min_margin: stats.min_margin,
            char_min_confidence: stats.char_min_confidence,
        })
    }
}

fn recognized_from_text_lines(lines: &mut [TextLine]) -> RecognizedText {
    recognized_from_text_lines_with_context(lines, None)
}

fn sort_and_truncate_by<T, F>(mut values: Vec<T>, max_count: usize, mut cmp: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    if values.len() <= 1 {
        values.sort_by(|a, b| cmp(a, b));
        return values;
    }
    if max_count == 0 {
        return Vec::new();
    }
    if values.len() > max_count {
        values.select_nth_unstable_by(max_count - 1, |a, b| cmp(a, b));
        values.truncate(max_count);
    }
    values.sort_by(|a, b| cmp(a, b));
    values
}

fn rec_candidate_cache_key(
    image: &DynamicImage,
    cfg: &OcrConfig,
    target_w: usize,
    variant: RecVariant,
) -> RecCandidateCacheKey {
    rec_candidate_cache_key_with_signature(
        dynamic_image_signature_cached(image),
        cfg.rec_img_h,
        cfg.rec_img_w,
        target_w,
        variant,
    )
}

fn rec_candidate_cache_key_with_signature(
    image_signature: u64,
    rec_img_h: usize,
    rec_img_w: usize,
    target_w: usize,
    variant: RecVariant,
) -> RecCandidateCacheKey {
    RecCandidateCacheKey {
        image_signature,
        target_w,
        rec_img_h,
        rec_img_w,
        variant: variant.into(),
    }
}

#[cfg(test)]
fn recognized_from_text_lines_with_image(
    lines: &mut [TextLine],
    image: &DynamicImage,
) -> RecognizedText {
    recognized_from_text_lines_with_context(lines, Some(image))
}

fn recognized_from_text_lines_with_context(
    lines: &mut [TextLine],
    image: Option<&DynamicImage>,
) -> RecognizedText {
    if lines.is_empty() {
        return RecognizedText::default();
    }

    lines.sort_by(reading_line_order);
    let deduped = dedupe_text_lines(lines);
    let filtered = filter_low_value_text_lines(&deduped);
    if filtered.is_empty() {
        return RecognizedText::default();
    }
    let rgb = image.map(to_rgb_on_white);
    let mut regions = if let Some(rgb) = rgb.as_ref() {
        group_text_lines_into_panel_regions(&filtered, rgb)
    } else {
        group_text_lines_into_regions(&filtered, None)
    };
    regions.sort_by(reading_region_order);

    let mut blocks = Vec::with_capacity(regions.len());
    let mut confidence_sum = 0.0f32;
    let mut confidence_weight_sum = 0.0f32;
    let mut public_regions = Vec::with_capacity(regions.len());
    for region in regions.iter_mut() {
        region.lines.sort_by(reading_line_order);
        let mut block = String::new();
        for line in region.lines.iter() {
            let text = line.text.trim();
            if text.is_empty() {
                continue;
            }
            if !block.is_empty() {
                block.push('\n');
            }
            block.push_str(text);
        }
        if block.trim().is_empty() {
            continue;
        }
        for line in &region.lines {
            let weight = line_confidence_weight(line);
            confidence_sum += line.confidence * weight;
            confidence_weight_sum += weight;
        }
        public_regions.push(public_region_from_layout(region, &block));
        blocks.push(block);
    }

    let text = blocks.join("\n\n");
    let line_count = text_line_count(&text);
    let region_count = blocks.len();
    let confidence = if confidence_weight_sum <= 0.0 {
        0.0
    } else {
        confidence_sum / confidence_weight_sum
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
            avg_margin: candidate.avg_margin,
            min_margin: candidate.min_margin,
            char_min_confidence: candidate.char_min_confidence,
            readable_ratio: readable_ratio(&text),
            support_count: 1,
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

fn candidate_text_lines(
    img: &DynamicImage,
    bbox: BoxRect,
    candidate: &RecCandidate,
    source: &str,
    transform: BboxTransform,
) -> Vec<TextLine> {
    let text = normalize_recognized_text(&candidate.text);
    let parts = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Vec::new();
    }

    if parts.len() == 1 {
        return vec![make_text_line(
            transform.map_box(bbox),
            text,
            candidate.confidence,
            candidate.avg_margin,
            candidate.min_margin,
            source.to_string(),
        )];
    }

    let line_boxes = split_recognized_multiline_box(img, bbox, parts.len());
    let source = format!("{source}:multiline");
    parts
        .into_iter()
        .zip(line_boxes)
        .map(|(part, line_box)| {
            make_text_line(
                transform.map_box(line_box),
                part.to_string(),
                candidate.confidence,
                candidate.avg_margin,
                candidate.min_margin,
                source.clone(),
            )
        })
        .collect()
}

fn public_region_from_layout(region: &LayoutRegion, text: &str) -> OcrTextRegion {
    let lines = region
        .lines
        .iter()
        .map(|line| OcrTextLine {
            bbox: box_to_array(line.bbox),
            text: line.text.clone(),
            confidence: line.confidence,
            avg_margin: line.avg_margin,
            min_margin: line.min_margin,
            char_min_confidence: line.char_min_confidence,
            readable_ratio: line.readable_ratio,
            support_count: line.support_count,
            source: line.source.clone(),
        })
        .collect::<Vec<_>>();
    let confidence = if region.lines.is_empty() {
        0.0
    } else {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for line in &region.lines {
            let weight = line_confidence_weight(line);
            sum += line.confidence * weight;
            weight_sum += weight;
        }
        if weight_sum <= 0.0 {
            0.0
        } else {
            sum / weight_sum
        }
    };
    OcrTextRegion {
        bbox: box_to_array(region.bbox),
        text: text.to_string(),
        confidence,
        source: dominant_region_source(&region.lines),
        lines,
    }
}

fn ocr_trace_lines_from_regions(regions: &[OcrTextRegion]) -> Vec<OcrTraceLine> {
    let mut trace_lines = Vec::new();
    for (region_idx, region) in regions.iter().enumerate() {
        if region.lines.is_empty() && !region.text.trim().is_empty() {
            let bbox = region.bbox;
            trace_lines.push(OcrTraceLine {
                region_index: region_idx,
                line_index: 0,
                bbox,
                crop_size: bbox_size(bbox),
                text: region.text.clone(),
                confidence: region.confidence,
                avg_margin: 0.0,
                min_margin: 0.0,
                char_min_confidence: region.confidence,
                readable_ratio: readable_ratio(&region.text),
                support_count: 1,
                source: region.source.clone(),
            });
            continue;
        }
        for (line_idx, line) in region.lines.iter().enumerate() {
            trace_lines.push(OcrTraceLine {
                region_index: region_idx,
                line_index: line_idx,
                bbox: line.bbox,
                crop_size: bbox_size(line.bbox),
                text: line.text.clone(),
                confidence: line.confidence,
                avg_margin: line.avg_margin,
                min_margin: line.min_margin,
                char_min_confidence: line.char_min_confidence,
                readable_ratio: line.readable_ratio,
                support_count: line.support_count,
                source: line.source.clone(),
            });
        }
    }
    trace_lines
}

fn bbox_size(bbox: [u32; 4]) -> [u32; 2] {
    [
        bbox[2].saturating_sub(bbox[0]),
        bbox[3].saturating_sub(bbox[1]),
    ]
}

#[allow(clippy::too_many_arguments)]
fn ocr_trace_json(
    image_w: u32,
    image_h: u32,
    source_has_alpha: bool,
    det_box_count: usize,
    color_region_count: usize,
    detect_used_whole_image_box: bool,
    empty_result: bool,
    confidence: f32,
    trace: &OcrTrace,
    regions: &[OcrTextRegion],
) -> String {
    let selected_source = trace.selected_source.as_deref().unwrap_or("");
    let mut out = String::new();
    out.push('{');
    out.push_str("\"image\":{");
    push_json_u32_field(&mut out, "width", image_w, false);
    push_json_u32_field(&mut out, "height", image_h, true);
    push_json_bool_field(&mut out, "source_has_alpha", source_has_alpha, true);
    out.push('}');

    out.push_str(",\"summary\":{");
    push_json_usize_field(&mut out, "det_box_count", det_box_count, false);
    push_json_usize_field(&mut out, "line_count", trace.lines.len(), true);
    push_json_usize_field(&mut out, "region_count", regions.len(), true);
    push_json_usize_field(&mut out, "color_region_count", color_region_count, true);
    push_json_usize_field(&mut out, "det_pass_count", trace.det_pass_count, true);
    push_json_usize_field(
        &mut out,
        "fallback_attempt_count",
        trace.fallback_attempt_count,
        true,
    );
    push_json_usize_field(
        &mut out,
        "rec_primary_call_count",
        trace.rec_primary_call_count,
        true,
    );
    push_json_usize_field(
        &mut out,
        "rec_alt_call_count",
        trace.rec_alt_call_count,
        true,
    );
    push_json_usize_field(&mut out, "candidate_count", trace.candidates.len(), true);
    push_json_usize_field(
        &mut out,
        "adopted_candidate_count",
        trace_candidate_action_count(&trace.candidates, "adopted"),
        true,
    );
    push_json_usize_field(
        &mut out,
        "rejected_candidate_count",
        trace_candidate_action_count(&trace.candidates, "rejected"),
        true,
    );
    push_json_usize_field(
        &mut out,
        "source_count",
        trace_source_family_count(&trace.lines),
        true,
    );
    push_json_bool_field(
        &mut out,
        "detect_used_whole_image_box",
        detect_used_whole_image_box,
        true,
    );
    push_json_bool_field(&mut out, "empty_result", empty_result, true);
    push_json_f32_field(&mut out, "confidence", confidence, true);
    push_json_str_field(&mut out, "selected_source", selected_source, true);
    out.push_str(",\"timing_ms\":{");
    push_json_u64_field(&mut out, "total", trace.timing.total_ms, false);
    push_json_u64_field(&mut out, "det", trace.timing.det_ms, true);
    push_json_u64_field(&mut out, "page_region", trace.timing.page_region_ms, true);
    push_json_u64_field(&mut out, "tile", trace.timing.tile_ms, true);
    push_json_u64_field(&mut out, "color_region", trace.timing.color_region_ms, true);
    push_json_u64_field(
        &mut out,
        "layered_region",
        trace.timing.layered_region_ms,
        true,
    );
    push_json_u64_field(
        &mut out,
        "visual_region",
        trace.timing.visual_region_ms,
        true,
    );
    push_json_u64_field(&mut out, "fallback", trace.timing.fallback_ms, true);
    push_json_u64_field(&mut out, "rec_primary", trace.timing.rec_primary_ms, true);
    push_json_u64_field(&mut out, "rec_alt", trace.timing.rec_alt_ms, true);
    push_json_u64_field(
        &mut out,
        "rec_cache_hit",
        trace.timing.rec_cache_hit_count,
        true,
    );
    push_json_u64_field(
        &mut out,
        "rec_cache_miss",
        trace.timing.rec_cache_miss_count,
        true,
    );
    push_json_u64_field(
        &mut out,
        "preprocess_call_count",
        trace.timing.preprocess_call_count,
        true,
    );
    push_json_u64_field(&mut out, "preprocess_ms", trace.timing.preprocess_ms, true);
    push_json_u64_field(
        &mut out,
        "variant_candidate_count",
        trace.timing.variant_candidate_count,
        true,
    );
    out.push('}');
    out.push('}');

    out.push_str(",\"regions\":[");
    for (region_idx, region) in regions.iter().enumerate() {
        if region_idx > 0 {
            out.push(',');
        }
        out.push('{');
        push_json_usize_field(&mut out, "index", region_idx, false);
        push_json_bbox_field(&mut out, "bbox", region.bbox, true);
        push_json_f32_field(&mut out, "confidence", region.confidence, true);
        push_json_str_field(&mut out, "source", &region.source, true);
        push_json_str_field(&mut out, "text", &region.text, true);
        out.push_str(",\"lines\":[");
        for (line_idx, line) in region.lines.iter().enumerate() {
            if line_idx > 0 {
                out.push(',');
            }
            out.push('{');
            push_json_usize_field(&mut out, "index", line_idx, false);
            push_json_bbox_field(&mut out, "bbox", line.bbox, true);
            push_json_size_field(&mut out, "crop_size", bbox_size(line.bbox), true);
            push_json_f32_field(&mut out, "confidence", line.confidence, true);
            push_json_f32_field(&mut out, "avg_margin", line.avg_margin, true);
            push_json_f32_field(&mut out, "min_margin", line.min_margin, true);
            push_json_f32_field(
                &mut out,
                "char_min_confidence",
                line.char_min_confidence,
                true,
            );
            push_json_f32_field(&mut out, "readable_ratio", line.readable_ratio, true);
            push_json_usize_field(&mut out, "support_count", line.support_count, true);
            push_json_str_field(&mut out, "source", &line.source, true);
            push_json_str_field(&mut out, "text", &line.text, true);
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push_str("],\"lines\":[");
    for (idx, line) in trace.lines.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        push_json_usize_field(&mut out, "region_index", line.region_index, false);
        push_json_usize_field(&mut out, "line_index", line.line_index, true);
        push_json_bbox_field(&mut out, "bbox", line.bbox, true);
        push_json_size_field(&mut out, "crop_size", line.crop_size, true);
        push_json_f32_field(&mut out, "confidence", line.confidence, true);
        push_json_f32_field(&mut out, "avg_margin", line.avg_margin, true);
        push_json_f32_field(&mut out, "min_margin", line.min_margin, true);
        push_json_f32_field(
            &mut out,
            "char_min_confidence",
            line.char_min_confidence,
            true,
        );
        push_json_f32_field(&mut out, "readable_ratio", line.readable_ratio, true);
        push_json_usize_field(&mut out, "support_count", line.support_count, true);
        push_json_str_field(&mut out, "source", &line.source, true);
        push_json_str_field(&mut out, "text", &line.text, true);
        out.push('}');
    }
    out.push_str("],\"candidates\":[");
    for (idx, candidate) in trace.candidates.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        push_json_usize_field(&mut out, "index", idx, false);
        push_json_str_field(&mut out, "label", &candidate.label, true);
        push_json_str_field(&mut out, "mode", &candidate.mode, true);
        push_json_str_field(&mut out, "action", &candidate.action, true);
        push_json_str_field(&mut out, "reason", &candidate.reason, true);
        push_json_f32_field(&mut out, "score", candidate.score, true);
        push_json_f32_field(&mut out, "confidence", candidate.confidence, true);
        push_json_usize_field(&mut out, "char_count", candidate.char_count, true);
        push_json_usize_field(&mut out, "line_count", candidate.line_count, true);
        push_json_usize_field(&mut out, "region_count", candidate.region_count, true);
        push_json_usize_field(
            &mut out,
            "source_family_count",
            candidate.source_family_count,
            true,
        );
        out.push('}');
    }
    out.push_str("]}");
    out
}

fn push_json_str_field(out: &mut String, key: &str, value: &str, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    out.push_str(&escape_json(value));
    out.push('"');
}

fn push_json_bool_field(out: &mut String, key: &str, value: bool, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(if value { "true" } else { "false" });
}

fn push_json_usize_field(out: &mut String, key: &str, value: usize, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
}

fn push_json_u32_field(out: &mut String, key: &str, value: u32, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
}

fn push_json_u64_field(out: &mut String, key: &str, value: u64, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&value.to_string());
}

fn push_json_f32_field(out: &mut String, key: &str, value: f32, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&format!("{value:.4}"));
}

fn push_json_bbox_field(out: &mut String, key: &str, bbox: [u32; 4], comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    out.push_str(&bbox[0].to_string());
    out.push(',');
    out.push_str(&bbox[1].to_string());
    out.push(',');
    out.push_str(&bbox[2].to_string());
    out.push(',');
    out.push_str(&bbox[3].to_string());
    out.push(']');
}

fn push_json_size_field(out: &mut String, key: &str, size: [u32; 2], comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":[");
    out.push_str(&size[0].to_string());
    out.push(',');
    out.push_str(&size[1].to_string());
    out.push(']');
}

fn escape_json(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn trace_candidate_action_count(candidates: &[OcrTraceCandidate], action: &str) -> usize {
    candidates
        .iter()
        .filter(|candidate| candidate.action == action)
        .count()
}

fn trace_source_family_count(lines: &[OcrTraceLine]) -> usize {
    let mut families = Vec::<&str>::new();
    for line in lines {
        let family = source_family(&line.source);
        if !families.contains(&family) {
            families.push(family);
        }
    }
    families.len()
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
        let duplicate = seen
            .iter()
            .any(|existing| text_keys_are_duplicates(existing, &key));
        if duplicate {
            continue;
        }
        seen.push(key);
        out.push(region.clone());
    }
    out
}

fn merge_recognized_line_sets(
    current_regions: &[OcrTextRegion],
    candidate: &RecognizedText,
) -> RecognizedText {
    let mut lines = text_lines_from_regions(current_regions);
    lines.extend(text_lines_from_recognized(candidate));
    recognized_from_text_lines(&mut lines)
}

fn text_lines_from_recognized(recognized: &RecognizedText) -> Vec<TextLine> {
    text_lines_from_regions(&recognized.regions)
}

fn text_lines_from_regions(regions: &[OcrTextRegion]) -> Vec<TextLine> {
    let mut lines = Vec::new();
    for region in regions {
        if region.lines.is_empty() {
            lines.push(make_text_line(
                box_from_array(region.bbox),
                region.text.clone(),
                region.confidence,
                0.0,
                0.0,
                region.source.clone(),
            ));
            continue;
        }
        for line in &region.lines {
            lines.push(make_text_line(
                box_from_array(line.bbox),
                line.text.clone(),
                line.confidence,
                line.avg_margin,
                line.min_margin,
                line.source.clone(),
            ));
        }
    }
    lines
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

fn group_text_lines_into_regions(
    lines: &[TextLine],
    rgb: Option<&image::RgbImage>,
) -> Vec<LayoutRegion> {
    let graph_regions = group_text_lines_into_graph_regions(lines, rgb);
    merge_layout_regions(graph_regions, rgb)
}

fn group_text_lines_into_graph_regions(
    lines: &[TextLine],
    rgb: Option<&image::RgbImage>,
) -> Vec<LayoutRegion> {
    let mut candidates = lines
        .iter()
        .filter(|line| !line.text.trim().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    candidates.sort_by(reading_line_order);
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut parents = (0..candidates.len()).collect::<Vec<_>>();
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            if candidates[j].bbox.1
                > candidates[i]
                    .bbox
                    .3
                    .saturating_add(GRAPH_REGION_VERTICAL_GAP_LIMIT)
            {
                break;
            }
            if line_graph_edge(&candidates[i], &candidates[j], rgb) {
                union_parent(&mut parents, i, j);
            }
        }
    }

    let mut groups: Vec<LayoutRegion> = Vec::with_capacity(candidates.len());
    let mut group_index: HashMap<usize, usize> = HashMap::with_capacity(candidates.len());
    for (idx, line) in candidates.into_iter().enumerate() {
        let root = find_parent(&mut parents, idx);
        if let Some(&group_idx) = group_index.get(&root) {
            groups[group_idx].add_line(line);
        } else {
            group_index.insert(root, groups.len());
            groups.push(LayoutRegion::from_line(line));
        }
    }

    for region in &mut groups {
        region.lines.sort_by(reading_line_order);
    }

    groups.sort_by(reading_region_order);
    groups
}

fn find_parent(parents: &mut [usize], idx: usize) -> usize {
    if parents[idx] != idx {
        let root = find_parent(parents, parents[idx]);
        parents[idx] = root;
    }
    parents[idx]
}

fn union_parent(parents: &mut [usize], a: usize, b: usize) {
    let root_a = find_parent(parents, a);
    let root_b = find_parent(parents, b);
    if root_a != root_b {
        parents[root_b] = root_a;
    }
}

fn line_graph_edge(a: &TextLine, b: &TextLine, rgb: Option<&image::RgbImage>) -> bool {
    let min_h = box_height(a.bbox).min(box_height(b.bbox)).max(1);
    let y_gap = vertical_gap(a.bbox, b.bbox);
    if y_gap > GRAPH_REGION_VERTICAL_GAP_LIMIT {
        return false;
    }
    let avg_h = (box_height(a.bbox).max(1) + box_height(b.bbox).max(1)) as f32 / 2.0;
    let y_overlap = vertical_overlap(a.bbox, b.bbox);
    let x_overlap = horizontal_overlap(a.bbox, b.bbox);
    let x_gap = horizontal_gap(a.bbox, b.bbox);
    let min_w = box_width(a.bbox).min(box_width(b.bbox)).max(1);
    let max_w = box_width(a.bbox).max(box_width(b.bbox)).max(1);
    let overlap_ratio = x_overlap as f32 / min_w as f32;
    let width_ratio = max_w as f32 / min_w as f32;
    let center_close =
        (box_center_x(a.bbox) - box_center_x(b.bbox)).abs() <= min_w.max(64) as f32 * 0.55;

    let same_visual_row = y_overlap.saturating_mul(100) >= min_h.saturating_mul(35)
        && x_gap <= min_h.saturating_mul(4).max(28);

    let can_merge = if same_visual_row && width_ratio <= 4.0 {
        true
    } else {
        if y_gap as f32 > (avg_h * 2.8).clamp(28.0, 96.0) {
            return false;
        }
        if width_ratio >= 1.85 && y_gap as f32 > avg_h * 1.2 {
            return false;
        }
        (overlap_ratio >= 0.42 || center_close) && width_ratio < 2.4
    };

    if !can_merge {
        return false;
    }

    if rgb.is_some_and(|rgb| visual_separator_between_boxes(rgb, a.bbox, b.bbox)) {
        return false;
    }

    true
}

fn group_text_lines_into_panel_regions(
    lines: &[TextLine],
    rgb: &image::RgbImage,
) -> Vec<LayoutRegion> {
    let panels = visual_page_region_boxes(&DynamicImage::ImageRgb8(rgb.clone()));
    if panels.len() < 2 {
        return group_text_lines_into_regions(lines, Some(rgb));
    }

    let mut panel_lines = vec![Vec::<TextLine>::new(); panels.len()];
    let mut unassigned = Vec::new();
    for line in lines.iter().filter(|line| !line.text.trim().is_empty()) {
        if let Some(idx) = best_panel_for_line(line.bbox, &panels) {
            panel_lines[idx].push(line.clone());
        } else {
            unassigned.push(line.clone());
        }
    }

    let mut regions = Vec::new();
    for mut bucket in panel_lines {
        if bucket.is_empty() {
            continue;
        }
        bucket.sort_by(reading_line_order);
        regions.extend(group_text_lines_into_regions(&bucket, Some(rgb)));
    }
    if !unassigned.is_empty() {
        unassigned.sort_by(reading_line_order);
        regions.extend(group_text_lines_into_regions(&unassigned, Some(rgb)));
    }
    regions
}

fn best_panel_for_line(line: BoxRect, panels: &[BoxRect]) -> Option<usize> {
    let line_area = box_area(line).max(1) as f32;
    let center = (box_center_x(line) as u32, (line.1 + line.3) / 2);
    let mut best = None;
    let mut best_score = 0.0f32;
    for (idx, panel) in panels.iter().enumerate() {
        let overlap = box_intersection_area(line, *panel) as f32 / line_area;
        let contains_center = point_in_box(center.0, center.1, *panel);
        let score = overlap + if contains_center { 0.35 } else { 0.0 };
        if score > best_score {
            best = Some(idx);
            best_score = score;
        }
    }
    if best_score >= 0.55 { best } else { None }
}

fn merge_layout_regions(
    regions: Vec<LayoutRegion>,
    rgb: Option<&image::RgbImage>,
) -> Vec<LayoutRegion> {
    if regions.len() <= 1 {
        return regions;
    }

    let max_region_height = regions
        .iter()
        .map(|region| box_height(region.bbox))
        .max()
        .unwrap_or(1);
    let avg_heights: Vec<f32> = regions.iter().map(region_average_line_height).collect();
    let max_merge_gap = ((max_region_height as f32) * 3.0).max(48.0);
    let mut sorted_indices: Vec<usize> = (0..regions.len()).collect();
    sorted_indices.sort_by(|&left, &right| {
        let a = regions[left].bbox;
        let b = regions[right].bbox;
        (a.1, a.0).cmp(&(b.1, b.0))
    });

    let mut parents = (0..regions.len()).collect::<Vec<_>>();
    for outer in 0..sorted_indices.len() {
        let i = sorted_indices[outer];
        for inner in (outer + 1)..sorted_indices.len() {
            let j = sorted_indices[inner];
            if regions[j].bbox.1
                > regions[i]
                    .bbox
                    .3
                    .saturating_add(max_merge_gap.ceil() as u32)
            {
                break;
            }
            if regions_should_merge(
                &regions[i],
                &regions[j],
                avg_heights[i],
                avg_heights[j],
                rgb,
            ) {
                union_parent(&mut parents, i, j);
            }
        }
    }

    let mut groups: Vec<LayoutRegion> = Vec::with_capacity(regions.len());
    let mut group_index: HashMap<usize, usize> = HashMap::with_capacity(regions.len());
    for (idx, mut region) in regions.into_iter().enumerate() {
        let root = find_parent(&mut parents, idx);
        if let Some(&group_idx) = group_index.get(&root) {
            let target: &mut LayoutRegion = &mut groups[group_idx];
            for line in region.lines.drain(..) {
                target.add_line(line);
            }
            continue;
        }

        group_index.insert(root, groups.len());
        groups.push(region);
    }

    groups.sort_by(reading_region_order);
    groups
}

fn regions_should_merge(
    a: &LayoutRegion,
    b: &LayoutRegion,
    avg_h_a: f32,
    avg_h_b: f32,
    rgb: Option<&image::RgbImage>,
) -> bool {
    let overlap = horizontal_overlap(a.bbox, b.bbox) as f32;
    let min_width = box_width(a.bbox).min(box_width(b.bbox)).max(1) as f32;
    let overlap_ratio = overlap / min_width;
    if overlap_ratio < 0.45 {
        return false;
    }
    let y_gap = vertical_gap(a.bbox, b.bbox) as f32;
    if y_gap > (avg_h_a.max(avg_h_b) * 3.0).max(48.0) {
        return false;
    }
    let width_ratio = box_width(a.bbox).max(box_width(b.bbox)).max(1) as f32
        / box_width(a.bbox).min(box_width(b.bbox)).max(1) as f32;
    if width_ratio >= 1.75 {
        return false;
    }

    if rgb.is_some_and(|rgb| visual_separator_between_boxes(rgb, a.bbox, b.bbox)) {
        return false;
    }

    true
}

fn reading_line_order(a: &TextLine, b: &TextLine) -> std::cmp::Ordering {
    (a.bbox.1 / 8, a.bbox.0).cmp(&(b.bbox.1 / 8, b.bbox.0))
}

fn reading_region_order(a: &LayoutRegion, b: &LayoutRegion) -> std::cmp::Ordering {
    reading_region_key(a).cmp(&reading_region_key(b))
}

fn reading_region_key(region: &LayoutRegion) -> (u32, u32, u32, u32, u32) {
    let bbox = region.bbox;
    let center_y = ((bbox.1 as u64 + bbox.3 as u64) / 2) as u32;
    let row_bucket = center_y / 32;
    (row_bucket, bbox.0, bbox.1, bbox.2, bbox.3)
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
            BboxTransform::Offset {
                dx,
                dy,
                max_w,
                max_h,
            } => clamp_box(
                (
                    b.0.saturating_add(dx),
                    b.1.saturating_add(dy),
                    b.2.saturating_add(dx),
                    b.3.saturating_add(dy),
                ),
                max_w,
                max_h,
            ),
            BboxTransform::ScaleOffset {
                sx,
                sy,
                dx,
                dy,
                max_w,
                max_h,
            } => {
                let x0 = ((b.0 as f32) * sx).floor().max(0.0) as u32;
                let y0 = ((b.1 as f32) * sy).floor().max(0.0) as u32;
                let x1 = ((b.2 as f32) * sx).ceil() as u32;
                let y1 = ((b.3 as f32) * sy).ceil() as u32;
                clamp_box(
                    (
                        x0.saturating_add(dx),
                        y0.saturating_add(dy),
                        x1.saturating_add(dx),
                        y1.saturating_add(dy),
                    ),
                    max_w,
                    max_h,
                )
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
        return binarize_low_contrast_foreground_from_rgb(&rgb);
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
        return binarize_low_contrast_foreground_from_rgb(&rgb);
    }
    Some(DynamicImage::ImageLuma8(out))
}

fn binarize_low_contrast_foreground_from_rgb(rgb: &image::RgbImage) -> Option<DynamicImage> {
    low_contrast_binary_luma_from_rgb(rgb).map(DynamicImage::ImageLuma8)
}

fn low_contrast_binary_luma_from_rgb(rgb: &image::RgbImage) -> Option<GrayImage> {
    let (w, h) = rgb.dimensions();
    if w < 8 || h < 6 {
        return None;
    }

    let gray = DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let local = local_binary_luma(&gray, false);
    if binary_foreground_is_text_like(&local) {
        return Some(local);
    }

    let stretched = contrast_stretch_luma(&gray);
    if stretched == gray {
        return None;
    }
    let local = local_binary_luma(&stretched, false);
    if binary_foreground_is_text_like(&local) {
        return Some(local);
    }
    None
}

fn binary_foreground_is_text_like(gray: &GrayImage) -> bool {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return false;
    }

    let mut count = 0usize;
    let mut min_x = w as usize;
    let mut min_y = h as usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            if gray.get_pixel(x as u32, y as u32)[0] >= 128 {
                continue;
            }
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    let total = (w as usize).saturating_mul(h as usize).max(1);
    let ratio = count as f32 / total as f32;
    if count < 4 || !(0.002..=0.38).contains(&ratio) {
        return false;
    }
    if max_x <= min_x || max_y <= min_y {
        return false;
    }
    let fg_w = max_x.saturating_sub(min_x).saturating_add(1);
    let fg_h = max_y.saturating_sub(min_y).saturating_add(1);
    if fg_w < 4 || fg_h < 2 {
        return false;
    }
    let mask = gray
        .pixels()
        .map(|pixel| pixel[0] < 128)
        .collect::<Vec<_>>();
    foreground_glyph_textness_score(&mask, w as usize, h as usize).is_some_and(|score| score >= -20)
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

fn push_recognition_candidate(
    candidate_pool: &mut Vec<OcrCandidateEntry>,
    label: String,
    candidate: &RecognizedText,
) -> bool {
    if candidate.text.trim().is_empty() {
        return false;
    }
    candidate_pool.push(OcrCandidateEntry {
        label,
        recognized: candidate.clone(),
    });
    true
}

fn recognized_from_candidate_pool(candidate_pool: &[OcrCandidateEntry]) -> RecognizedText {
    recognized_from_candidate_pool_with_context(candidate_pool, None)
}

fn recognized_from_candidate_pool_with_image(
    candidate_pool: &[OcrCandidateEntry],
    image: &DynamicImage,
) -> RecognizedText {
    recognized_from_candidate_pool_with_context(candidate_pool, Some(image))
}

fn recognized_from_candidate_pool_with_context(
    candidate_pool: &[OcrCandidateEntry],
    image: Option<&DynamicImage>,
) -> RecognizedText {
    let mut lines = Vec::new();
    for entry in candidate_pool {
        let mut entry_lines = text_lines_from_recognized(&entry.recognized);
        for line in &mut entry_lines {
            if line.source.trim().is_empty() || line.source == "unknown" {
                line.source = entry.label.clone();
            }
        }
        lines.extend(entry_lines);
    }
    recognized_from_text_lines_with_context(&mut lines, image)
}

#[allow(clippy::too_many_arguments)]
fn maybe_adopt_candidate_pool(
    text: &mut String,
    confidence: &mut f32,
    line_count: &mut usize,
    region_count: &mut usize,
    layout_applied: &mut bool,
    regions: &mut Vec<OcrTextRegion>,
    fallback: &mut Option<String>,
    candidate_pool: &mut Vec<OcrCandidateEntry>,
    image: Option<&DynamicImage>,
    label: String,
    candidate: &RecognizedText,
) -> bool {
    if !push_recognition_candidate(candidate_pool, label.clone(), candidate) {
        return false;
    }
    let pooled = if let Some(image) = image {
        recognized_from_candidate_pool_with_image(candidate_pool, image)
    } else {
        recognized_from_candidate_pool(candidate_pool)
    };
    if !pooled_recognition_is_better(text, *confidence, *line_count, &pooled) {
        return false;
    }

    *text = pooled.text;
    *confidence = pooled.confidence;
    *line_count = pooled.line_count.max(text_line_count(text));
    *region_count = pooled
        .region_count
        .max(if text.trim().is_empty() { 0 } else { 1 });
    *layout_applied = pooled.layout_applied;
    *regions = pooled.regions;
    *fallback = Some(format!("pooled:{label}"));
    true
}

#[allow(clippy::too_many_arguments)]
fn maybe_adopt_candidate_pool_traced(
    text: &mut String,
    confidence: &mut f32,
    line_count: &mut usize,
    region_count: &mut usize,
    layout_applied: &mut bool,
    regions: &mut Vec<OcrTextRegion>,
    fallback: &mut Option<String>,
    candidate_pool: &mut Vec<OcrCandidateEntry>,
    image: Option<&DynamicImage>,
    trace: &mut OcrTrace,
    label: String,
    candidate: &RecognizedText,
) -> bool {
    let before_text = text.clone();
    let before_confidence = *confidence;
    let before_line_count = *line_count;
    let before_pool_len = candidate_pool.len();
    let adopted = maybe_adopt_candidate_pool(
        text,
        confidence,
        line_count,
        region_count,
        layout_applied,
        regions,
        fallback,
        candidate_pool,
        image,
        label.clone(),
        candidate,
    );
    if candidate_pool.len() == before_pool_len {
        trace.candidates.push(trace_candidate_event(
            label, "pool", "ignored", "empty", candidate,
        ));
        return adopted;
    }

    let pooled = if let Some(image) = image {
        recognized_from_candidate_pool_with_image(candidate_pool, image)
    } else {
        recognized_from_candidate_pool(candidate_pool)
    };
    let reason = pooled_candidate_decision_reason(
        &before_text,
        before_confidence,
        before_line_count,
        &pooled,
        adopted,
    );
    trace.candidates.push(trace_candidate_event(
        label,
        "pool",
        if adopted { "adopted" } else { "rejected" },
        reason,
        &pooled,
    ));
    adopted
}

fn pooled_recognition_is_better(
    current_text: &str,
    current_confidence: f32,
    current_line_count: usize,
    pooled: &RecognizedText,
) -> bool {
    let pooled_chars = recognized_char_count(&pooled.text);
    if pooled_chars == 0 {
        return false;
    }
    let current_chars = recognized_char_count(current_text);
    if current_chars == 0 {
        return true;
    }

    let source_support = pooled_source_family_count(pooled);
    if pooled.line_count > current_line_count
        && pooled_chars >= current_chars
        && (pooled.confidence + 0.12 >= current_confidence || source_support >= 3)
    {
        return true;
    }
    if pooled_chars > current_chars + 2
        && (pooled.confidence + 0.16 >= current_confidence
            || pooled.confidence >= 0.42
            || source_support >= 3)
    {
        return true;
    }
    if pooled.confidence > current_confidence + 0.06 && pooled_chars + 2 >= current_chars {
        return true;
    }

    let current_quality = recognition_text_quality(current_text, current_confidence);
    let pooled_quality = recognized_text_quality_score(pooled)
        + pooled.line_count.saturating_sub(current_line_count) as f32 * 1.5
        + source_support.saturating_sub(1) as f32 * 2.0;
    pooled_quality > current_quality + 5.0 && pooled_chars + 1 >= current_chars
}

fn recognized_text_quality_score(recognized: &RecognizedText) -> f32 {
    let lines = text_lines_from_recognized(recognized);
    if lines.is_empty() {
        return recognition_text_quality(&recognized.text, recognized.confidence);
    }
    let quality_sum = lines.iter().map(text_line_quality).sum::<f32>();
    let avg_quality = quality_sum / lines.len() as f32;
    let chars = recognized_char_count(&recognized.text) as f32;
    avg_quality + chars.min(80.0) * 0.12 + recognized.region_count.saturating_sub(1) as f32 * 0.8
        - recognized
            .text
            .lines()
            .filter(|line| is_low_value_short_ocr_line(line))
            .count() as f32
            * 2.0
}

fn pooled_source_family_count(pooled: &RecognizedText) -> usize {
    let mut families: Vec<String> = Vec::new();
    for line in text_lines_from_recognized(pooled) {
        let family = source_family(&line.source).to_string();
        if !families.contains(&family) {
            families.push(family);
        }
    }
    families.len()
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

    if !regions.is_empty() && !candidate.regions.is_empty() {
        let merged = merge_recognized_line_sets(regions, candidate);
        let merged_chars = recognized_char_count(&merged.text);
        if merged_chars > current_chars + 2 && candidate.confidence + 0.10 >= *confidence {
            *text = merged.text;
            *confidence = merged.confidence;
            *line_count = merged.line_count.max(text_line_count(text));
            *region_count = merged
                .region_count
                .max(if text.trim().is_empty() { 0 } else { 1 });
            *layout_applied = merged.layout_applied;
            *regions = merged.regions;
            *fallback = Some(format!("merged:{label}"));
            return true;
        }
    } else {
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

#[allow(clippy::too_many_arguments)]
fn maybe_adopt_recognized_traced(
    text: &mut String,
    confidence: &mut f32,
    line_count: &mut usize,
    region_count: &mut usize,
    layout_applied: &mut bool,
    regions: &mut Vec<OcrTextRegion>,
    fallback: &mut Option<String>,
    trace: &mut OcrTrace,
    label: String,
    candidate: &RecognizedText,
) -> bool {
    let before_text = text.clone();
    let before_confidence = *confidence;
    let before_line_count = *line_count;
    let adopted = maybe_adopt_recognized(
        text,
        confidence,
        line_count,
        region_count,
        layout_applied,
        regions,
        fallback,
        label.clone(),
        candidate,
    );
    let reason = single_candidate_decision_reason(
        &before_text,
        before_confidence,
        before_line_count,
        candidate,
        adopted,
    );
    trace.candidates.push(trace_candidate_event(
        label,
        "single",
        if adopted { "adopted" } else { "rejected" },
        reason,
        candidate,
    ));
    adopted
}

fn trace_candidate_event(
    label: String,
    mode: &str,
    action: &str,
    reason: &str,
    candidate: &RecognizedText,
) -> OcrTraceCandidate {
    OcrTraceCandidate {
        label,
        mode: mode.to_string(),
        action: action.to_string(),
        reason: reason.to_string(),
        score: if candidate.text.trim().is_empty() {
            0.0
        } else {
            recognized_text_quality_score(candidate)
        },
        confidence: candidate.confidence,
        char_count: recognized_char_count(&candidate.text),
        line_count: candidate.line_count,
        region_count: candidate.region_count,
        source_family_count: pooled_source_family_count(candidate),
    }
}

fn single_candidate_decision_reason(
    current_text: &str,
    current_confidence: f32,
    current_line_count: usize,
    candidate: &RecognizedText,
    adopted: bool,
) -> &'static str {
    if candidate.text.trim().is_empty() {
        return "empty";
    }
    if adopted {
        return "adopted";
    }
    candidate_rejection_reason(
        current_text,
        current_confidence,
        current_line_count,
        candidate,
    )
}

fn pooled_candidate_decision_reason(
    current_text: &str,
    current_confidence: f32,
    current_line_count: usize,
    pooled: &RecognizedText,
    adopted: bool,
) -> &'static str {
    if pooled.text.trim().is_empty() {
        return "empty-pooled";
    }
    if adopted {
        return "adopted";
    }
    candidate_rejection_reason(current_text, current_confidence, current_line_count, pooled)
}

fn candidate_rejection_reason(
    current_text: &str,
    current_confidence: f32,
    current_line_count: usize,
    candidate: &RecognizedText,
) -> &'static str {
    let current_chars = recognized_char_count(current_text);
    let candidate_chars = recognized_char_count(&candidate.text);
    if candidate_chars == 0 {
        return "empty";
    }
    if current_chars == 0 {
        return "current-empty";
    }
    if candidate.line_count > current_line_count && candidate_chars >= current_chars {
        return "line-gain-threshold";
    }
    if candidate_chars <= current_chars + 2 && candidate.confidence <= current_confidence + 0.08 {
        return "not-better";
    }
    if candidate.confidence + 0.16 < current_confidence {
        return "lower-confidence";
    }
    "quality-threshold"
}

#[cfg(test)]
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
            lines.push(make_text_line(
                bbox,
                line.text.clone(),
                line.confidence,
                0.0,
                0.0,
                line.source.clone(),
            ));
        }
    }

    recognized_from_text_lines(&mut lines)
}

fn filter_page_region_supplement(
    candidate: &RecognizedText,
    existing_regions: &[OcrTextRegion],
) -> RecognizedText {
    let existing_boxes = collect_region_line_boxes(existing_regions);
    if existing_boxes.is_empty() {
        return candidate.clone();
    }

    let existing_keys = collect_region_text_keys(existing_regions);
    let mut lines = Vec::new();
    for region in &candidate.regions {
        for line in &region.lines {
            if !is_strong_page_region_supplement(line) {
                continue;
            }
            if !existing_keys.is_empty()
                && page_region_line_is_existing_fragment(&line.text, &existing_keys)
            {
                continue;
            }
            let bbox = box_from_array(line.bbox);
            if existing_boxes
                .iter()
                .any(|existing| boxes_significantly_overlap(bbox, *existing))
            {
                continue;
            }
            lines.push(make_text_line(
                bbox,
                line.text.clone(),
                line.confidence,
                0.0,
                0.0,
                line.source.clone(),
            ));
        }
    }

    recognized_from_text_lines(&mut lines)
}

fn filter_color_region_det_supplement(
    candidate: &RecognizedText,
    existing_regions: &[OcrTextRegion],
) -> RecognizedText {
    let existing_boxes = collect_region_line_boxes(existing_regions);
    let existing_keys = collect_region_text_keys(existing_regions);
    let mut lines = Vec::new();
    for region in &candidate.regions {
        for line in &region.lines {
            if !is_strong_page_region_supplement(line) {
                continue;
            }
            if page_region_line_is_existing_fragment(&line.text, &existing_keys) {
                continue;
            }
            let bbox = box_from_array(line.bbox);
            if existing_boxes
                .iter()
                .any(|existing| boxes_significantly_overlap(bbox, *existing))
            {
                continue;
            }
            lines.push(make_text_line(
                bbox,
                line.text.clone(),
                line.confidence,
                0.0,
                0.0,
                line.source.clone(),
            ));
        }
    }

    recognized_from_text_lines(&mut lines)
}

fn collect_region_text_keys(regions: &[OcrTextRegion]) -> Vec<String> {
    regions
        .iter()
        .flat_map(|region| {
            if region.lines.is_empty() {
                vec![normalize_ocr_line(&region.text)]
            } else {
                region
                    .lines
                    .iter()
                    .map(|line| normalize_ocr_line(&line.text))
                    .collect::<Vec<_>>()
            }
        })
        .filter(|key| !key.is_empty())
        .collect()
}

fn page_region_line_is_existing_fragment(text: &str, existing_keys: &[String]) -> bool {
    let text = text.trim();
    let key = normalize_ocr_line(text);
    if key.is_empty() {
        return true;
    }
    if text.ends_with('@') || text.ends_with(':') || text.ends_with('：') {
        return true;
    }
    let key_chars = key.chars().count();
    if key_chars <= 4 && !text.chars().any(is_cjk_char) {
        return existing_keys
            .iter()
            .any(|existing| existing.len() > key.len() && existing.contains(&key));
    }
    false
}

fn is_strong_page_region_supplement(line: &OcrTextLine) -> bool {
    let text = line.text.trim();
    let chars = recognized_char_count(text);
    if chars == 0 || line.confidence < 0.45 {
        return false;
    }
    if chars < 3 && !(chars >= 2 && text.chars().any(is_cjk_char)) {
        return false;
    }
    readable_ratio(text) >= 0.60
        && dominant_char_ratio(text) < 0.70
        && punctuation_ratio(text) < 0.65
        && !is_low_value_short_ocr_line(text)
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
    let area = box_intersection_area(a, b);
    if area == 0 {
        return false;
    }
    let min_area = box_area(a).min(box_area(b)).max(1);
    area as f32 / min_area as f32 >= 0.50
}

#[derive(Copy, Clone)]
struct SupplementSeenBox {
    bbox: BoxRect,
    has_reliable_text: bool,
}

struct SupplementSeenIndex {
    buckets: Vec<Vec<usize>>,
    boxes: Vec<BoxRect>,
    bucket_count: usize,
}

impl SupplementSeenIndex {
    fn new(image_height: u32) -> Self {
        let bucket_count =
            (image_height as usize / BOX_DEDUPE_BUCKET_SIZE as usize).saturating_add(1);
        Self {
            buckets: vec![Vec::new(); bucket_count.max(1)],
            boxes: Vec::new(),
            bucket_count: bucket_count.max(1),
        }
    }

    fn insert_if_reliable(&mut self, b: BoxRect, has_reliable_text: bool) {
        if !has_reliable_text {
            return;
        }
        let idx = self.boxes.len();
        self.boxes.push(b);
        let (start_bucket, end_bucket) = box_bucket_range(b, self.bucket_count);
        for bucket_idx in start_bucket..=end_bucket {
            self.buckets[bucket_idx].push(idx);
        }
    }

    fn is_redundant(&self, candidate: BoxRect) -> bool {
        if self.boxes.is_empty() {
            return false;
        }
        let (start_bucket, end_bucket) = box_bucket_range(candidate, self.bucket_count);
        let candidate_area = box_area(candidate).max(1);
        for bucket_idx in start_bucket..=end_bucket {
            for &idx in &self.buckets[bucket_idx] {
                if idx >= self.boxes.len() {
                    continue;
                }
                let seen_box = self.boxes[idx];
                let overlap = box_intersection_area(candidate, seen_box);
                if overlap == 0 {
                    continue;
                }
                let seen_area = box_area(seen_box).max(1);
                if overlap as f32 / candidate_area as f32 >= 0.90
                    || overlap as f32 / seen_area as f32 >= 0.90
                {
                    return true;
                }
            }
        }
        false
    }
}

fn supplement_box_is_redundant(candidate: BoxRect, seen_boxes: &[SupplementSeenBox]) -> bool {
    let candidate_area = box_area(candidate).max(1);
    seen_boxes.iter().any(|seen| {
        if !seen.has_reliable_text {
            return false;
        }
        let overlap = box_intersection_area(candidate, seen.bbox);
        if overlap == 0 {
            return false;
        }
        let seen_area = box_area(seen.bbox).max(1);
        overlap as f32 / candidate_area as f32 >= 0.90 || overlap as f32 / seen_area as f32 >= 0.90
    })
}

fn supplement_lines_are_reliable(lines: &[TextLine]) -> bool {
    if lines.is_empty() {
        return false;
    }
    let total_lines = lines.len();
    lines.iter().any(|line| {
        let text = line.text.trim();
        !text.is_empty()
            && !is_low_value_short_ocr_line(text)
            && !is_low_value_text_line(line, total_lines)
            && line.confidence >= MIN_ACCEPT_REC_CONFIDENCE
            && readable_ratio(text) >= 0.45
    })
}

fn box_intersection_area(a: BoxRect, b: BoxRect) -> u64 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    if x1 <= x0 || y1 <= y0 {
        return 0;
    }
    box_area((x0, y0, x1, y1))
}

fn dedupe_text_lines(lines: &[TextLine]) -> Vec<TextLine> {
    let mut clusters: Vec<Vec<TextLine>> = Vec::with_capacity(lines.len());
    'candidate: for line in lines.iter().filter(|line| !line.text.trim().is_empty()) {
        for cluster in &mut clusters {
            if cluster
                .iter()
                .any(|cluster_line| text_lines_are_near_duplicates(cluster_line, line))
            {
                cluster.push(line.clone());
                continue 'candidate;
            }
        }
        clusters.push(vec![line.clone()]);
    }

    let mut kept = clusters
        .iter()
        .filter_map(|cluster| select_voted_text_line(cluster))
        .collect::<Vec<_>>();
    kept.sort_by(reading_line_order);
    kept
}

fn select_voted_text_line(cluster: &[TextLine]) -> Option<TextLine> {
    if cluster.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_score = voted_text_line_quality(&cluster[best_idx], cluster);
    for (idx, line) in cluster.iter().enumerate().skip(1) {
        let score = voted_text_line_quality(line, cluster);
        if score > best_score + 0.01
            || ((score - best_score).abs() <= 0.01
                && text_line_quality(line) > text_line_quality(&cluster[best_idx]))
        {
            best_idx = idx;
            best_score = score;
        }
    }
    let mut selected = cluster[best_idx].clone();
    let selected_key = normalize_ocr_line(&selected.text);
    if !selected_key.is_empty() {
        let mut support_count = 1usize;
        for other in cluster {
            if std::ptr::eq(other, &cluster[best_idx]) {
                continue;
            }
            if normalize_ocr_line(&other.text) == selected_key
                || normalized_text_similarity(&selected.text, &other.text) >= 0.90
            {
                support_count += 1;
            }
        }
        selected.support_count = support_count;
    } else {
        selected.support_count = cluster.len().max(1);
    }
    Some(selected)
}

fn voted_text_line_quality(line: &TextLine, cluster: &[TextLine]) -> f32 {
    let key = normalize_ocr_line(&line.text);
    if key.is_empty() {
        return text_line_quality(line);
    }

    let mut exact_support = 0usize;
    let mut near_support = 0usize;
    let mut source_families: Vec<&str> = vec![source_family(&line.source)];
    for other in cluster {
        if std::ptr::eq(other, line) {
            continue;
        }
        let similarity = normalized_text_similarity(&line.text, &other.text);
        if normalize_ocr_line(&other.text) == key {
            exact_support += 1;
        } else if similarity >= 0.90 {
            near_support += 1;
        }
        if similarity >= 0.82 {
            let family = source_family(&other.source);
            if !source_families.contains(&family) {
                source_families.push(family);
            }
        }
    }

    text_line_quality(line)
        + exact_support as f32 * 24.0
        + near_support as f32 * 3.0
        + source_families.len().saturating_sub(1) as f32 * 4.0
}

fn source_family(source: &str) -> &str {
    source.split(':').next().unwrap_or(source)
}

fn text_lines_are_near_duplicates(a: &TextLine, b: &TextLine) -> bool {
    let similarity = normalized_text_similarity(&a.text, &b.text);
    if long_texts_are_near_duplicates(&a.text, &b.text) && similarity >= 0.96 {
        return true;
    }
    if boxes_significantly_overlap(a.bbox, b.bbox) || box_iou(a.bbox, b.bbox) >= 0.35 {
        return similarity >= 0.78;
    }
    if long_texts_are_near_duplicates(&a.text, &b.text)
        && boxes_share_column(a.bbox, b.bbox)
        && similarity >= 0.90
    {
        return true;
    }
    long_texts_are_near_duplicates(&a.text, &b.text)
        && boxes_are_near_same_column(a.bbox, b.bbox)
        && similarity >= 0.88
}

fn text_line_quality(line: &TextLine) -> f32 {
    let text = line.text.trim();
    let chars = recognized_char_count(text) as f32;
    line.confidence * 100.0
        + line.char_min_confidence.clamp(0.0, 1.0) * 10.0
        + line.avg_margin.clamp(0.0, 1.0) * 8.0
        + line.min_margin.clamp(0.0, 1.0) * 3.0
        + line.readable_ratio * 12.0
        + chars.min(32.0) * 0.35
        + line.support_count.saturating_sub(1) as f32 * 5.0
        + source_quality_bonus(&line.source)
        - punctuation_ratio(text) * 8.0
        - dominant_char_ratio(text) * 4.0
}

fn line_confidence_weight(line: &TextLine) -> f32 {
    let text = line.text.trim();
    let chars = recognized_char_count(text) as f32;
    let margin_bonus =
        (line.avg_margin.clamp(0.0, 1.0) * 1.4 + line.min_margin.clamp(0.0, 1.0) * 0.6).min(1.2);
    let readability = line.readable_ratio.clamp(0.0, 1.0);
    let source = (source_quality_bonus(&line.source) / 8.0).clamp(-0.20, 0.50);
    let support = line.support_count.saturating_sub(1).min(4) as f32 * 0.12;
    let char_floor = line.char_min_confidence.clamp(0.0, 1.0) * 0.35;
    (0.45
        + chars.min(24.0) / 24.0
        + readability * 0.55
        + margin_bonus
        + source
        + support
        + char_floor)
        .clamp(0.25, 3.4)
}

fn source_quality_bonus(source: &str) -> f32 {
    let family = source_family(source);
    if family == "det" {
        4.0
    } else if family == "tile-region" || family == "page-region" {
        2.5
    } else if family == "color-region-det" || family == "layered-region" {
        1.5
    } else if family == "visual-region" || family == "color-region" {
        0.5
    } else if family == "line-crops" {
        -1.0
    } else {
        0.0
    }
}

fn normalized_text_similarity(a: &str, b: &str) -> f32 {
    let a = normalize_ocr_line(a);
    let b = normalize_ocr_line(b);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }
    let min_len = a.chars().count().min(b.chars().count());
    let max_len = a.chars().count().max(b.chars().count()).max(1);
    if min_len >= 4 && (a.contains(&b) || b.contains(&a)) {
        return min_len as f32 / max_len as f32;
    }
    let distance = levenshtein_chars(&a, &b);
    1.0 - distance as f32 / max_len as f32
}

fn text_keys_are_duplicates(existing: &str, key: &str) -> bool {
    if existing == key {
        return true;
    }
    let min_len = existing.chars().count().min(key.chars().count());
    let max_len = existing.chars().count().max(key.chars().count()).max(1);
    if min_len >= 4
        && (existing.contains(key) || key.contains(existing))
        && min_len as f32 / max_len as f32 >= 0.72
    {
        return true;
    }
    min_len >= 8 && normalized_text_similarity(existing, key) >= 0.88
}

fn long_texts_are_near_duplicates(a: &str, b: &str) -> bool {
    normalize_ocr_line(a)
        .chars()
        .count()
        .min(normalize_ocr_line(b).chars().count())
        >= 8
}

fn filter_low_value_text_lines(lines: &[TextLine]) -> Vec<TextLine> {
    if lines.len() < 6 {
        return lines.to_vec();
    }
    lines
        .iter()
        .filter(|line| !is_low_value_text_line(line, lines.len()))
        .cloned()
        .collect()
}

fn is_low_value_text_line(line: &TextLine, total_lines: usize) -> bool {
    let text = line.text.trim();
    if is_low_value_short_ocr_line(text) {
        return true;
    }
    if total_lines < 10 {
        return false;
    }
    if line.confidence < 0.68 && is_low_value_ascii_noise(text) {
        return true;
    }
    let has_margin = line.avg_margin > 0.0 || line.min_margin > 0.0;
    has_margin && line.confidence < 0.72 && line.avg_margin < 0.025 && readable_ratio(text) < 0.75
}

fn is_low_value_ascii_noise(text: &str) -> bool {
    let chars = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    if chars.len() < 3 || chars.len() > 18 {
        return false;
    }
    if !chars.iter().all(|ch| ch.is_ascii_alphabetic()) {
        return false;
    }
    let uppercase_after_first = chars
        .iter()
        .skip(1)
        .filter(|ch| ch.is_ascii_uppercase())
        .count();
    let lowercase = chars.iter().filter(|ch| ch.is_ascii_lowercase()).count();
    chars.len() <= 3 || (uppercase_after_first > 0 && lowercase > 0)
}

fn is_low_value_short_ocr_line(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() {
        return true;
    }
    if text.chars().any(is_cjk_char) {
        return false;
    }
    let chars = text
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<Vec<_>>();
    if chars.is_empty() {
        return true;
    }
    if chars.iter().all(|ch| ch.is_ascii_punctuation()) {
        return true;
    }
    if chars.len() <= 1 && chars.iter().all(|ch| ch.is_ascii()) {
        return true;
    }
    chars.len() <= 2
        && chars.iter().all(|ch| ch.is_ascii())
        && !chars.iter().any(|ch| ch.is_ascii_lowercase())
}

fn is_cjk_char(ch: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&ch)
}

fn boxes_share_column(a: BoxRect, b: BoxRect) -> bool {
    let min_width = box_width(a).min(box_width(b)).max(1) as f32;
    let overlap_ratio = horizontal_overlap(a, b) as f32 / min_width;
    let center_close = (box_center_x(a) - box_center_x(b)).abs()
        <= box_width(a).max(box_width(b)).max(1) as f32 * 0.45;
    overlap_ratio >= 0.35 || center_close
}

fn boxes_are_near_same_column(a: BoxRect, b: BoxRect) -> bool {
    let max_h = box_height(a).max(box_height(b)).max(1);
    let max_gap = (max_h * 10).clamp(80, 220);
    boxes_share_column(a, b) && vertical_gap(a, b) <= max_gap
}

fn levenshtein_chars(a: &str, b: &str) -> usize {
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
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
        let duplicate = seen
            .iter()
            .any(|existing| text_keys_are_duplicates(existing, &key));
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

fn should_use_forced_split_lines(
    bbox: BoxRect,
    direct: Option<&RecCandidate>,
    split_lines: &[TextLine],
) -> bool {
    if should_use_split_lines(direct, split_lines) {
        return true;
    }

    let Some(direct) = direct else {
        return false;
    };
    let direct_text = normalize_recognized_text(&direct.text);
    if !recognized_box_needs_repair(bbox, &direct_text, direct.confidence) {
        return false;
    }

    let split_text = split_lines
        .iter()
        .map(|line| line.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let split_chars = recognized_char_count(&split_text);
    if split_chars < 4 || readable_ratio(&split_text) < 0.60 {
        return false;
    }

    let split_confidence = split_lines.iter().map(|line| line.confidence).sum::<f32>()
        / split_lines.len().max(1) as f32;
    let split_quality = recognition_text_quality(&split_text, split_confidence);
    let direct_quality = recognition_text_quality(&direct_text, direct.confidence);
    split_quality > direct_quality + 6.0
}

fn consume_recognition_budget(
    count: usize,
    split_line_rec_budget: &mut usize,
    line_repair_rec_budget: &mut usize,
) {
    let split_used = (*split_line_rec_budget).min(count);
    *split_line_rec_budget = (*split_line_rec_budget).saturating_sub(split_used);
    *line_repair_rec_budget = (*line_repair_rec_budget).saturating_sub(count - split_used);
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

fn quality_fallback_family_budget(text: &str) -> usize {
    if text.trim().is_empty() {
        MAX_QUALITY_FALLBACK_FAMILIES_EMPTY
    } else {
        MAX_QUALITY_FALLBACK_FAMILIES_PARTIAL
    }
}

fn quality_fallback_enhancement_variant_budget(
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
) -> usize {
    if text.trim().is_empty() {
        return MAX_QUALITY_FALLBACK_ENHANCEMENT_VARIANTS;
    }
    if confidence <= 0.0 || confidence < 0.40 {
        return MAX_QUALITY_FALLBACK_ENHANCEMENT_VARIANTS;
    }
    if confidence < 0.58 || det_box_count >= 6 || line_count == 0 {
        return (MAX_QUALITY_FALLBACK_ENHANCEMENT_VARIANTS - 1).max(1);
    }
    if line_count >= det_box_count.max(1) {
        return 4;
    }
    MAX_ENHANCEMENT_VARIANTS_PER_PASS
}

fn consume_quality_fallback_family(budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    true
}

fn should_enhance_crop(b: BoxRect) -> bool {
    box_width(b) <= 480 && box_height(b) <= 96 && box_area(b) <= 48_000
}

fn recognized_box_needs_repair(b: BoxRect, text: &str, confidence: f32) -> bool {
    let chars = recognized_char_count(text);
    if chars == 0 {
        return true;
    }
    if confidence < MIN_STRONG_REC_CONFIDENCE {
        return true;
    }
    if chars >= 4
        && (readable_ratio(text) < 0.65
            || dominant_char_ratio(text) >= 0.72
            || punctuation_ratio(text) >= 0.62)
    {
        return true;
    }

    let w = box_width(b);
    let h = box_height(b).max(1);
    let aspect = w as f32 / h as f32;
    (w >= 480 || h >= 72 || aspect >= 16.0) && confidence < 0.75
}

fn repair_candidate_is_better(current: Option<&RecCandidate>, candidate: &RecCandidate) -> bool {
    let Some(current) = current else {
        return true;
    };
    let current_text = normalize_recognized_text(&current.text);
    let candidate_text = normalize_recognized_text(&candidate.text);
    let current_chars = recognized_char_count(&current_text);
    let candidate_chars = recognized_char_count(&candidate_text);
    if candidate_chars == 0 {
        return false;
    }
    if current_chars == 0 {
        return true;
    }

    let current_quality = recognition_candidate_model_score(current);
    let candidate_quality = recognition_candidate_model_score(candidate);
    candidate_quality > current_quality + 4.0
        || (candidate_chars > current_chars + 2
            && candidate.confidence + 0.12 >= current.confidence)
}

fn recognition_candidate_is_better(
    current: Option<&RecCandidate>,
    candidate: &RecCandidate,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    let current_text = normalize_recognized_text(&current.text);
    let candidate_text = normalize_recognized_text(&candidate.text);
    let current_chars = recognized_char_count(&current_text);
    let candidate_chars = recognized_char_count(&candidate_text);
    if candidate_chars == 0 {
        return false;
    }
    if current_chars == 0 {
        return true;
    }

    let current_quality = recognition_candidate_model_score(current);
    let candidate_quality = recognition_candidate_model_score(candidate);
    candidate_quality > current_quality + 2.0
        || (candidate_chars > current_chars + 1
            && candidate.confidence + 0.08 >= current.confidence)
}

fn local_recognition_candidate_is_final(candidate: &RecCandidate) -> bool {
    if !is_usable_recognition(candidate) {
        return false;
    }
    if candidate.confidence < 0.92 {
        return false;
    }
    if candidate.avg_margin < 0.12 || candidate.min_margin < 0.04 {
        return false;
    }
    if candidate.char_min_confidence < 0.92 {
        return false;
    }
    let text = normalize_ocr_line(&candidate.text);
    if text.chars().count() <= 1 {
        return false;
    }
    readable_ratio(&candidate.text) >= 0.90
}

fn should_try_alt_recognition(primary: &RecCandidate) -> bool {
    let text = normalize_recognized_text(&primary.text);
    if text.is_empty() {
        return true;
    }
    if !is_usable_recognition(primary) {
        return true;
    }
    if is_stable_short_text_candidate(primary, &text, MAX_FAST_TEXT_CHARS) {
        return false;
    }
    let ascii = ascii_ratio(&text);
    if ascii >= 0.35 {
        return true;
    }
    if primary.confidence < 0.58 {
        return true;
    }
    if primary.avg_margin < 0.05 || primary.min_margin < 0.02 {
        return true;
    }
    false
}

fn join_segment_recognition_text(segments: &[RecCandidate]) -> String {
    let mut out = String::new();
    for segment in segments {
        let text = normalize_recognized_text(&segment.text);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(overlap) = segment_text_overlap(&out, text) {
            out.push_str(&text.chars().skip(overlap).collect::<String>());
            continue;
        }
        if should_insert_segment_space(&out, text) {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

fn segment_text_overlap(left: &str, right: &str) -> Option<usize> {
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    let max_overlap = left_chars.len().min(right_chars.len()).min(12);
    for len in (2..=max_overlap).rev() {
        let left_tail = left_chars[left_chars.len() - len..]
            .iter()
            .collect::<String>();
        let right_head = right_chars[..len].iter().collect::<String>();
        if normalize_ocr_line(&left_tail) == normalize_ocr_line(&right_head) {
            return Some(len);
        }
    }
    None
}

fn should_insert_segment_space(left: &str, right: &str) -> bool {
    let Some(left_ch) = left.chars().rev().find(|ch| !ch.is_whitespace()) else {
        return false;
    };
    let Some(right_ch) = right.chars().find(|ch| !ch.is_whitespace()) else {
        return false;
    };
    left_ch.is_ascii_alphanumeric() && right_ch.is_ascii_alphanumeric()
}

fn recognition_text_quality(text: &str, confidence: f32) -> f32 {
    recognition_text_quality_with_margin(text, confidence, 0.0, 0.0)
}

fn recognition_text_quality_with_margin(
    text: &str,
    confidence: f32,
    avg_margin: f32,
    min_margin: f32,
) -> f32 {
    let chars = recognized_char_count(text) as f32;
    confidence * 100.0
        + readable_ratio(text) * 14.0
        + chars.min(32.0) * 0.25
        + avg_margin.clamp(0.0, 1.0) * 10.0
        + min_margin.clamp(0.0, 1.0) * 4.0
        - punctuation_ratio(text) * 9.0
        - dominant_char_ratio(text) * 5.0
}

fn prioritize_supplement_candidate_boxes(
    image: &DynamicImage,
    boxes: Vec<BoxRect>,
    existing_regions: &[OcrTextRegion],
) -> Vec<BoxRect> {
    let focus_band = primary_content_focus_band(image, existing_regions);
    let mut scored = boxes
        .into_iter()
        .filter(|b| !color_region_box_covered_by_reliable_text(*b, existing_regions))
        .filter_map(|b| supplemental_text_candidate_score(image, b).map(|score| (b, score)))
        .collect::<Vec<_>>();
    if scored.len() > MAX_SUPPLEMENT_CANDIDATE_SCORE_PRESELECT {
        let keep = MAX_SUPPLEMENT_CANDIDATE_SCORE_PRESELECT.max(1);
        scored.select_nth_unstable_by(keep - 1, |a, b| {
            supplement_candidate_score_cmp(a, b, focus_band)
        });
        scored.truncate(keep);
    }
    sort_supplement_candidate_scores(&mut scored, focus_band);
    scored = retain_focus_prioritized_candidates(scored, focus_band);
    scored.into_iter().map(|(b, _)| b).collect()
}

fn primary_content_focus_band(
    image: &DynamicImage,
    existing_regions: &[OcrTextRegion],
) -> Option<(u32, u32)> {
    if existing_regions.len() < 2 {
        return None;
    }
    let (img_w, _) = image.dimensions();
    if img_w < 240 {
        return None;
    }

    let mut buckets: Vec<(u32, u64)> = Vec::new();
    let mut samples: Vec<(BoxRect, u32)> = Vec::new();
    for region in existing_regions {
        if region.lines.is_empty() {
            let b = box_from_array(region.bbox);
            let bucket = supplement_focus_bucket(b);
            let score = supplement_focus_sample_score(b);
            if score > 0 {
                buckets.push((bucket, score));
                samples.push((b, bucket));
            }
            continue;
        }
        for line in &region.lines {
            let b = box_from_array(line.bbox);
            let bucket = supplement_focus_bucket(b);
            let score = supplement_focus_sample_score(b);
            if score > 0 {
                buckets.push((bucket, score));
                samples.push((b, bucket));
            }
        }
    }
    if samples.len() < 2 {
        return None;
    }

    let mut bucket_scores: Vec<(u32, u64)> = Vec::new();
    for (bucket, score) in buckets {
        if let Some((_, total)) = bucket_scores.iter_mut().find(|(seen, _)| *seen == bucket) {
            *total += score;
        } else {
            bucket_scores.push((bucket, score));
        }
    }
    let Some((best_bucket, _)) = bucket_scores.into_iter().max_by_key(|(_, score)| *score) else {
        return None;
    };

    let mut focus_boxes = Vec::new();
    let mut wide_count = 0usize;
    for (b, bucket) in samples {
        if bucket.abs_diff(best_bucket) > 1 {
            continue;
        }
        if box_width(b) >= box_height(b).saturating_mul(3).max(72) {
            wide_count += 1;
        }
        focus_boxes.push(b);
    }
    if focus_boxes.len() < 2 || wide_count == 0 {
        return None;
    }

    let mut left = img_w;
    let mut right = 0u32;
    for b in &focus_boxes {
        left = left.min(b.0);
        right = right.max(b.2);
    }
    if right <= left {
        return None;
    }
    let band_w = right.saturating_sub(left);
    if band_w < (img_w / 6).max(96) {
        return None;
    }
    let pad = (band_w / 6).max(48);
    Some((
        left.saturating_sub(pad),
        right.saturating_add(pad).min(img_w),
    ))
}

fn supplement_focus_bucket(b: BoxRect) -> u32 {
    (box_center_x(b) / 96.0).floor() as u32
}

fn supplement_focus_sample_score(b: BoxRect) -> u64 {
    let w = box_width(b);
    let h = box_height(b);
    if w < 32 || h < 8 {
        return 0;
    }
    let area = box_area(b).min(24_000) as u64;
    let wide_bonus = if w >= h.saturating_mul(3).max(72) {
        area
    } else {
        area / 4
    };
    area + wide_bonus + w.min(600) as u64 * 24
}

fn supplement_focus_rank(b: BoxRect, focus_band: Option<(u32, u32)>) -> u8 {
    if let Some((focus_left, focus_right)) = focus_band {
        if box_intersects_focus_band(b, focus_left, focus_right) {
            return 1;
        }
    }
    0
}

fn sort_supplement_candidate_scores(scored: &mut [(BoxRect, i64)], focus_band: Option<(u32, u32)>) {
    scored.sort_by(|a, b| supplement_candidate_score_cmp(a, b, focus_band));
}

fn supplement_candidate_score_cmp(
    a: &(BoxRect, i64),
    b: &(BoxRect, i64),
    focus_band: Option<(u32, u32)>,
) -> std::cmp::Ordering {
    supplement_focus_rank(b.0, focus_band)
        .cmp(&supplement_focus_rank(a.0, focus_band))
        .then_with(|| b.1.cmp(&a.1))
        .then_with(|| reading_box_order(&a.0, &b.0))
}

fn retain_focus_prioritized_candidates(
    scored: Vec<(BoxRect, i64)>,
    focus_band: Option<(u32, u32)>,
) -> Vec<(BoxRect, i64)> {
    let Some((focus_left, focus_right)) = focus_band else {
        return scored;
    };
    let focus_count = scored
        .iter()
        .filter(|(b, _)| box_intersects_focus_band(*b, focus_left, focus_right))
        .count();
    if focus_count < 2 {
        return scored;
    }

    let mut outside_count = 0usize;
    scored
        .into_iter()
        .filter(|(b, _)| {
            if box_intersects_focus_band(*b, focus_left, focus_right) {
                return true;
            }
            if outside_count >= MAX_SUPPLEMENT_OUTSIDE_FOCUS_CANDIDATES {
                return false;
            }
            outside_count += 1;
            true
        })
        .collect()
}

fn box_intersects_focus_band(b: BoxRect, focus_left: u32, focus_right: u32) -> bool {
    if horizontal_overlap(b, (focus_left, b.1, focus_right, b.3)) > 0 {
        return true;
    }
    let center_x = box_center_x(b);
    center_x >= focus_left as f32 && center_x <= focus_right as f32
}

fn supplemental_text_candidate_score(image: &DynamicImage, b: BoxRect) -> Option<i64> {
    if !supplement_box_is_worth_recognition(b) {
        return None;
    }
    let crop = crop_box(image, b);
    let rgb = to_rgb_on_white(&crop);
    let (w, h) = rgb.dimensions();
    let mask = text_foreground_mask_from_rgb(&rgb)?;
    let Some((min_x, min_y, max_x, max_y, foreground_count)) =
        foreground_bounds_from_mask(&mask, w as usize, h as usize)
    else {
        return None;
    };
    let area = box_area(b).max(1);
    let foreground_ratio = foreground_count as f32 / area as f32;
    if !(0.001..=0.55).contains(&foreground_ratio) {
        return None;
    }

    let text_w = max_x.saturating_sub(min_x).saturating_add(1);
    let text_h = max_y.saturating_sub(min_y).saturating_add(1);
    if text_w < 8 || text_h < 4 {
        return None;
    }
    let approx_slots = text_w as f32 / text_h.max(1) as f32;
    if area < 1_600 && (text_w < 24 || approx_slots < 1.6) {
        return None;
    }
    let glyph_score = foreground_glyph_textness_score(&mask, w as usize, h as usize)?;
    if glyph_score < -20 {
        return None;
    }

    let density_score = (foreground_ratio * 12_000.0).round() as i64;
    let foreground_score = (foreground_count.min(4000) / 4) as i64;
    let extent_score = (text_w + text_h).min(1200) as i64 / 2;
    let split_bonus = if large_text_box_needs_structured_split(b) {
        24
    } else {
        0
    };
    let area_penalty = (area / 12_000) as i64;
    Some(density_score + foreground_score + extent_score + split_bonus + glyph_score - area_penalty)
}

fn layered_color_region_text_boxes(
    image: &DynamicImage,
    existing_regions: &[OcrTextRegion],
) -> Vec<BoxRect> {
    let mut panels = color_region_boxes(image);
    panels.extend(visual_page_region_boxes(image));
    panels = nms_boxes(panels, 0.85);
    if panels.is_empty() {
        return Vec::new();
    }

    let mut candidates: Vec<BoxRect> = Vec::new();
    for panel in panels {
        if box_width(panel) < 32 || box_height(panel) < 16 {
            continue;
        }
        if color_region_det_box_covered_by_reliable_text(panel, existing_regions)
            && !large_text_box_needs_structured_split(panel)
        {
            continue;
        }
        candidates.extend(layered_text_boxes_in_panel(image, panel, 0, 16));
    }
    candidates = nms_boxes(candidates, 0.55);
    candidates = dedupe_box_candidates(candidates);
    let mut candidates = prioritize_supplement_candidate_boxes(image, candidates, existing_regions);
    candidates.truncate(MAX_COLOR_REGION_CANDIDATES);
    candidates
}

fn layered_text_boxes_in_panel(
    image: &DynamicImage,
    panel: BoxRect,
    depth: usize,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if max_boxes == 0 || box_width(panel) < 16 || box_height(panel) < 8 {
        return Vec::new();
    }

    let crop = crop_box(image, panel);
    let (img_w, img_h) = image.dimensions();
    let mut boxes = foreground_line_boxes(&crop, max_boxes)
        .into_iter()
        .map(|b| offset_local_box(panel, b, img_w, img_h))
        .filter(|b| layered_text_line_box_is_plausible(panel, *b))
        .collect::<Vec<_>>();
    if boxes.len() < max_boxes {
        let remaining = max_boxes.saturating_sub(boxes.len());
        boxes.extend(
            foreground_component_text_boxes(&crop, remaining)
                .into_iter()
                .map(|b| offset_local_box(panel, b, img_w, img_h))
                .filter(|b| layered_text_line_box_is_plausible(panel, *b)),
        );
        boxes = nms_boxes(boxes, 0.60);
    }

    if boxes.len() < max_boxes {
        let remaining = max_boxes.saturating_sub(boxes.len());
        boxes.extend(
            dominant_color_layer_text_boxes(&crop, remaining)
                .into_iter()
                .map(|b| offset_local_box(panel, b, img_w, img_h))
                .filter(|b| layered_text_line_box_is_plausible(panel, *b)),
        );
        boxes = nms_boxes(boxes, 0.58);
    }

    if depth < 2 && boxes.len() < max_boxes {
        let crop_area = box_area(image_box(&crop)).max(1);
        for child in color_region_boxes(&crop) {
            if box_area(child).saturating_mul(100) >= crop_area.saturating_mul(88) {
                continue;
            }
            let child = offset_local_box(panel, child, img_w, img_h);
            if box_width(child) < 24 || box_height(child) < 12 {
                continue;
            }
            let remaining = max_boxes.saturating_sub(boxes.len());
            if remaining == 0 {
                break;
            }
            boxes.extend(layered_text_boxes_in_panel(
                image,
                child,
                depth + 1,
                remaining,
            ));
        }
    }

    boxes = sort_and_truncate_by(boxes, max_boxes, reading_box_order);
    boxes
}

fn dominant_color_layer_text_boxes(image: &DynamicImage, max_boxes: usize) -> Vec<BoxRect> {
    if max_boxes == 0 {
        return Vec::new();
    }
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(mask) = soft_color_foreground_mask_from_rgb(&rgb) else {
        return Vec::new();
    };
    let mut boxes = line_boxes_from_foreground_mask(&mask, w as usize, h as usize, max_boxes);
    if boxes.len() < max_boxes {
        boxes.extend(component_line_boxes_from_mask(
            &mask,
            w as usize,
            h as usize,
            max_boxes.saturating_sub(boxes.len()),
        ));
        boxes = nms_boxes(boxes, 0.60);
    }
    boxes = sort_and_truncate_by(boxes, max_boxes, reading_box_order);
    boxes
}

fn offset_local_box(parent: BoxRect, local: BoxRect, max_w: u32, max_h: u32) -> BoxRect {
    clamp_box(
        (
            parent.0.saturating_add(local.0),
            parent.1.saturating_add(local.1),
            parent.0.saturating_add(local.2),
            parent.1.saturating_add(local.3),
        ),
        max_w,
        max_h,
    )
}

fn layered_text_line_box_is_plausible(panel: BoxRect, b: BoxRect) -> bool {
    if box_width(b) < 8 || box_height(b) < 4 {
        return false;
    }
    if box_area(b).saturating_mul(100) >= box_area(panel).max(1).saturating_mul(90) {
        return false;
    }
    box_height(b).saturating_mul(100) <= box_height(panel).max(1).saturating_mul(75)
}

fn foreground_component_text_boxes(image: &DynamicImage, max_boxes: usize) -> Vec<BoxRect> {
    if max_boxes == 0 {
        return Vec::new();
    }
    let rgb = to_rgb_on_white(image);
    let (w_u32, h_u32) = rgb.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(mask) = text_foreground_mask_from_rgb(&rgb) else {
        return Vec::new();
    };
    component_line_boxes_from_mask(&mask, w, h, max_boxes)
}

fn component_line_boxes_from_mask(
    mask: &[bool],
    w: usize,
    h: usize,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if max_boxes == 0 || w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return Vec::new();
    }
    let mut visited = vec![false; mask.len()];
    let mut components = Vec::new();
    for idx in 0..mask.len() {
        if visited[idx] || !mask[idx] {
            continue;
        }
        let mut stack = vec![idx];
        visited[idx] = true;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut count = 0usize;
        while let Some(cur) = stack.pop() {
            let x = cur % w;
            let y = cur / w;
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx >= w || ny >= h {
                    continue;
                }
                let nidx = ny * w + nx;
                if visited[nidx] || !mask[nidx] {
                    continue;
                }
                visited[nidx] = true;
                stack.push(nidx);
            }
        }
        if count < 3 || max_x <= min_x || max_y <= min_y {
            continue;
        }
        let b = (
            min_x.saturating_sub(1) as u32,
            min_y.saturating_sub(1) as u32,
            (max_x + 2).min(w) as u32,
            (max_y + 2).min(h) as u32,
        );
        if component_box_is_text_like(b, count) {
            components.push(b);
        }
    }
    let mut merged = merge_component_boxes_into_lines(components);
    merged.retain(|b| box_width(*b) >= 8 && box_height(*b) >= 4);
    sort_and_truncate_by(merged, max_boxes, reading_box_order)
}

fn component_box_is_text_like(b: BoxRect, foreground_count: usize) -> bool {
    let w = box_width(b);
    let h = box_height(b);
    if w < 2 || h < 2 {
        return false;
    }
    if w > 220 || h > 96 {
        return false;
    }
    let area = box_area(b).max(1);
    let density = foreground_count as f32 / area as f32;
    (0.04..=0.80).contains(&density) && w.saturating_mul(6) >= h && h.saturating_mul(20) >= w
}

fn foreground_glyph_textness_score(mask: &[bool], w: usize, h: usize) -> Option<i64> {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return None;
    }
    let mut visited = vec![false; mask.len()];
    let mut component_count = 0usize;
    let mut text_like_count = 0usize;
    let mut largest_density = 0.0f32;
    let mut largest_area = 0u64;
    let mut largest_box = (0u32, 0u32);
    for idx in 0..mask.len() {
        if visited[idx] || !mask[idx] {
            continue;
        }
        let mut stack = vec![idx];
        visited[idx] = true;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut count = 0usize;
        while let Some(cur) = stack.pop() {
            let x = cur % w;
            let y = cur / w;
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx >= w || ny >= h {
                    continue;
                }
                let nidx = ny * w + nx;
                if visited[nidx] || !mask[nidx] {
                    continue;
                }
                visited[nidx] = true;
                stack.push(nidx);
            }
        }
        if count < 2 || max_x <= min_x || max_y <= min_y {
            continue;
        }
        component_count += 1;
        let b = (
            min_x as u32,
            min_y as u32,
            (max_x + 1).min(w) as u32,
            (max_y + 1).min(h) as u32,
        );
        let area = box_area(b).max(1);
        let density = count as f32 / area as f32;
        if area > largest_area {
            largest_area = area;
            largest_density = density;
            largest_box = (box_width(b), box_height(b));
        }
        if component_box_is_text_like(b, count)
            || (box_width(b) >= 5
                && box_height(b) >= 2
                && box_width(b).saturating_mul(16) >= box_height(b)
                && (0.02..=0.88).contains(&density))
            || (box_width(b) >= box_height(b).saturating_mul(4)
                && box_height(b) >= 2
                && density <= 1.0)
        {
            text_like_count += 1;
        }
    }
    if component_count == 0 {
        return None;
    }

    let mut score = text_like_count as i64 * 12 + component_count.min(12) as i64 * 2;
    if text_like_count == 0 {
        score -= 24;
    }
    if component_count <= 1
        && largest_density > 0.86
        && largest_box.0 < largest_box.1.saturating_mul(4)
    {
        score -= 32;
    }
    if component_count >= 3 {
        score += 10;
    }
    Some(score.clamp(-48, 72))
}

fn merge_component_boxes_into_lines(mut boxes: Vec<BoxRect>) -> Vec<BoxRect> {
    boxes.sort_by(reading_box_order);
    let mut merged: Vec<BoxRect> = Vec::new();
    for b in boxes {
        if let Some(last) = merged.last_mut() {
            let y_overlap = vertical_overlap(*last, b);
            let min_h = box_height(*last).min(box_height(b)).max(1);
            let gap = horizontal_gap(*last, b);
            if y_overlap.saturating_mul(100) >= min_h.saturating_mul(35)
                && gap <= min_h.saturating_mul(3).max(12)
            {
                *last = union_box(*last, b);
                continue;
            }
        }
        merged.push(b);
    }
    merged
}

fn color_region_det_candidate_boxes(
    image: &DynamicImage,
    existing_regions: &[OcrTextRegion],
    max_boxes: usize,
) -> (usize, Vec<BoxRect>) {
    let boxes = color_region_boxes(image);
    let total = boxes.len();
    if max_boxes == 0 || boxes.is_empty() {
        return (total, Vec::new());
    }

    let mut scored = boxes
        .into_iter()
        .filter(|b| box_width(*b) >= 48 && box_height(*b) >= 16)
        .filter(|b| !color_region_det_box_covered_by_reliable_text(*b, existing_regions))
        .filter_map(|b| supplemental_text_candidate_score(image, b).map(|score| (b, score)))
        .collect::<Vec<_>>();
    if scored.is_empty() {
        return (total, Vec::new());
    }
    scored = retain_top_scored_candidates(scored, max_boxes);
    let mut boxes = scored.into_iter().map(|(b, _)| b).collect::<Vec<_>>();
    boxes = dedupe_box_candidates(boxes);
    boxes = sort_and_truncate_by(boxes, max_boxes, reading_box_order);
    (total, boxes)
}

fn color_region_det_box_covered_by_reliable_text(b: BoxRect, regions: &[OcrTextRegion]) -> bool {
    let mut reliable_count = 0usize;
    let mut max_line_h = 0u32;
    let mut max_line_w = 0u32;
    for region in regions {
        if region.lines.is_empty() {
            let line_box = box_from_array(region.bbox);
            if boxes_significantly_overlap(b, line_box)
                && !recognized_box_needs_repair(line_box, &region.text, region.confidence)
            {
                reliable_count += 1;
                max_line_h = max_line_h.max(box_height(line_box));
                max_line_w = max_line_w.max(box_width(line_box));
            }
            continue;
        }

        for line in &region.lines {
            let line_box = box_from_array(line.bbox);
            if boxes_significantly_overlap(b, line_box)
                && !recognized_box_needs_repair(line_box, &line.text, line.confidence)
            {
                reliable_count += 1;
                max_line_h = max_line_h.max(box_height(line_box));
                max_line_w = max_line_w.max(box_width(line_box));
            }
        }
    }
    if reliable_count == 0 {
        return false;
    }

    let line_like_h = max_line_h.saturating_mul(3).max(36);
    let line_like_w = max_line_w.saturating_mul(4).max(160);
    box_height(b) <= line_like_h && box_width(b) <= line_like_w
}

fn color_region_box_covered_by_reliable_text(b: BoxRect, regions: &[OcrTextRegion]) -> bool {
    for region in regions {
        if region.lines.is_empty() {
            let line_box = box_from_array(region.bbox);
            if boxes_significantly_overlap(b, line_box)
                && !recognized_box_needs_repair(line_box, &region.text, region.confidence)
            {
                return true;
            }
            continue;
        }
        for line in &region.lines {
            let line_box = box_from_array(line.bbox);
            if boxes_significantly_overlap(b, line_box)
                && !recognized_box_needs_repair(line_box, &line.text, line.confidence)
            {
                return true;
            }
        }
    }
    false
}

fn split_line_recognition_budget(source: &str) -> usize {
    if source == "det" {
        MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS
    } else if source.starts_with("page-region:") {
        MAX_PAGE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS
    } else if source.starts_with("color-region-det:") {
        MAX_COLOR_REGION_DET_SPLIT_LINE_RECOGNITIONS_PER_PASS
    } else if source.starts_with("tile-region:") {
        MAX_TILE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS
    } else {
        0
    }
}

fn line_repair_recognition_budget(source: &str) -> usize {
    if source == "det" {
        MAX_LINE_REPAIR_RECOGNITIONS_PER_PASS
    } else if source.starts_with("page-region:") {
        MAX_PAGE_REGION_REPAIR_RECOGNITIONS_PER_PASS
    } else if source.starts_with("color-region-det:") {
        MAX_COLOR_REGION_DET_REPAIR_RECOGNITIONS_PER_PASS
    } else if source.starts_with("tile-region:") {
        MAX_TILE_REGION_REPAIR_RECOGNITIONS_PER_PASS
    } else {
        0
    }
}

fn should_use_high_res_tile_supplement(
    img: &DynamicImage,
    cfg: &OcrConfig,
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
    regions: &[OcrTextRegion],
) -> bool {
    let (w, h) = img.dimensions();
    if w <= cfg.det_img_side as u32 && h <= cfg.det_img_side as u32 {
        return false;
    }
    if line_count == 0 {
        return true;
    }
    if confidence > 0.0 && confidence < 0.62 {
        return true;
    }
    if det_box_count >= 10 && line_count * 2 <= det_box_count {
        return true;
    }
    if recognized_char_count(text) < 8 && det_box_count >= 4 {
        return true;
    }
    regions_have_repairable_lines(regions)
}

fn should_use_uncovered_visual_supplement(
    img: &DynamicImage,
    cfg: &OcrConfig,
    text: &str,
    det_box_count: usize,
    line_count: usize,
    regions: &[OcrTextRegion],
) -> bool {
    if regions.is_empty() {
        return det_box_count == 0 || line_count == 0;
    }
    let (w, h) = img.dimensions();
    if (w >= cfg.det_img_side as u32 || h >= cfg.det_img_side as u32)
        && (line_count == 0 || recognized_char_count(text) < 8)
    {
        return true;
    }
    if det_box_count >= 6 && line_count * 3 <= det_box_count * 2 {
        return true;
    }
    regions_have_repairable_lines(regions)
}

fn should_use_eager_color_region_supplement(
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
    regions: &[OcrTextRegion],
) -> bool {
    if text.trim().is_empty() {
        return true;
    }
    if confidence > 0.0 && confidence < 0.58 {
        return true;
    }
    if det_box_count >= 6 && line_count * 3 <= det_box_count * 2 {
        return true;
    }
    if recognized_char_count(text) < 8 && det_box_count >= 3 {
        return true;
    }
    regions_have_repairable_lines(regions)
}

fn should_continue_eager_supplements(
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
    regions: &[OcrTextRegion],
) -> bool {
    if text.trim().is_empty() {
        return true;
    }
    let stable_hint = normalize_recognized_text(text);
    let stable_chars = recognized_char_count(&stable_hint);
    if confidence >= 0.95 && stable_chars >= 6 && !regions_have_repairable_lines(regions) {
        return false;
    }
    if confidence > 0.0 && confidence < 0.60 {
        return true;
    }
    if det_box_count >= 6 && line_count * 3 <= det_box_count * 2 {
        return true;
    }
    if recognized_char_count(text) < 10 && det_box_count >= 3 {
        return true;
    }
    regions_have_repairable_lines(regions)
}

fn should_continue_eager_supplement_pass(
    text: &str,
    confidence: f32,
    det_box_count: usize,
    line_count: usize,
    regions: &[OcrTextRegion],
    previous_text: &str,
    previous_confidence: f32,
    previous_line_count: usize,
) -> bool {
    should_continue_eager_supplements(text, confidence, det_box_count, line_count, regions)
        && (supplement_pass_made_progress(
            previous_text,
            previous_confidence,
            previous_line_count,
            text,
            confidence,
            line_count,
        ) || regions_have_repairable_lines(regions))
}

fn supplement_pass_made_progress(
    previous_text: &str,
    previous_confidence: f32,
    previous_line_count: usize,
    current_text: &str,
    current_confidence: f32,
    current_line_count: usize,
) -> bool {
    let previous_chars = recognized_char_count(previous_text);
    let current_chars = recognized_char_count(current_text);
    if current_chars >= previous_chars + MIN_SUPPLEMENT_CHAR_GROWTH {
        return true;
    }
    if current_line_count > previous_line_count {
        return true;
    }
    let gain_requirement = if previous_chars <= 6 || current_chars <= 6 {
        MIN_SUPPLEMENT_CONFIDENCE_GAIN.min(0.02)
    } else {
        MIN_SUPPLEMENT_CONFIDENCE_GAIN
    };
    if current_confidence >= previous_confidence + gain_requirement {
        return true;
    }
    false
}

fn supplement_box_is_worth_recognition(b: BoxRect) -> bool {
    let w = box_width(b);
    let h = box_height(b).max(1);
    let area = box_area(b);
    if area >= 6_000 && w >= 120 && h >= 28 {
        return true;
    }
    if area < 700 && (w < 56 || h < 16) {
        return false;
    }
    if w < 28 || h < 8 {
        return false;
    }
    let aspect = w as f32 / h as f32;
    if aspect < 1.2 && area < 2_400 {
        return false;
    }
    if h > 96 && w < h.saturating_mul(2) {
        return false;
    }
    true
}

fn regions_have_repairable_lines(regions: &[OcrTextRegion]) -> bool {
    for region in regions {
        if region.lines.is_empty() {
            if recognized_box_needs_repair(
                box_from_array(region.bbox),
                &region.text,
                region.confidence,
            ) {
                return true;
            }
            continue;
        }
        for line in &region.lines {
            if recognized_box_needs_repair(box_from_array(line.bbox), &line.text, line.confidence) {
                return true;
            }
        }
    }
    false
}

fn recognized_char_count(text: &str) -> usize {
    text.chars().filter(|ch| !ch.is_whitespace()).count()
}

fn text_line_count(text: &str) -> usize {
    text.lines().filter(|line| !line.trim().is_empty()).count()
}

fn should_short_circuit_crop_enhancement(
    best: Option<&RecCandidate>,
    stale_streak: usize,
    variant_budget: usize,
) -> bool {
    if stale_streak < 2 || variant_budget < 3 {
        return false;
    }

    let Some(best) = best else {
        return false;
    };
    if best.confidence >= 0.96 {
        return true;
    }

    let normalized_text = normalize_recognized_text(&best.text);
    let len = normalized_text.chars().count();
    if best.confidence >= 0.90 && len >= 6 {
        return true;
    }
    best.confidence >= 0.84 && len >= 16
}

type BoxRect = (u32, u32, u32, u32);

impl OrtOcrEngine {
    fn detect_text_boxes(
        &self,
        img: &DynamicImage,
        cfg: &OcrConfig,
        include_raw_split_candidates: bool,
    ) -> Result<Vec<DetectionBox>, String> {
        let (det_input, det_shape, sx, sy, src_w, src_h) =
            preprocess_det_image(img, cfg.det_img_side)?;
        let det_input = [det_input.as_slice()];
        let det_shape = [det_shape.as_slice()];
        let (det_output, _det_shapes) = ort::run_session(&self.det, &det_input, &det_shape)?;
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

        let scaled = nms_boxes(scaled, 0.35);
        let source_rgb = to_rgb_on_white(img);
        let mut merged = merge_nearby_detection_boxes(&source_rgb, scaled.clone());
        merged = normalize_detection_boxes(merged);

        merged.retain(|(x0, y0, x1, y1)| x1 > x0 && y1 > y0);
        merged.sort_by(reading_box_order);
        let boxes = merged
            .into_iter()
            .map(|bbox| DetectionBox {
                alternatives: if include_raw_split_candidates {
                    dedupe_box_candidates(raw_split_detection_candidates(bbox, &scaled))
                } else {
                    Vec::new()
                },
                bbox,
            })
            .collect();
        Ok(boxes)
    }
}

fn merge_nearby_detection_boxes(rgb: &image::RgbImage, mut boxes: Vec<BoxRect>) -> Vec<BoxRect> {
    boxes.sort_by(reading_box_order);
    let mut merged: Vec<BoxRect> = Vec::new();
    for b in boxes {
        if let Some(last) = merged.last_mut() {
            let y_overlap = vertical_overlap(*last, b);
            let x_gap = if b.0 > last.2 { b.0 - last.2 } else { 0 };
            let line_h = (last.3 - last.1).min(b.3 - b.1);
            let new_w = last.2.max(b.2) - last.0.min(b.0);
            let enough_row_overlap = y_overlap.saturating_mul(100) >= line_h.saturating_mul(45);
            let max_gap = if new_w >= 720 {
                (line_h * 2).max(16)
            } else {
                (line_h * 3).max(20)
            };
            if enough_row_overlap
                && x_gap <= max_gap
                && new_w <= 1200
                && !boxes_have_merge_separator(rgb, *last, b)
            {
                last.0 = last.0.min(b.0);
                last.1 = last.1.min(b.1);
                last.2 = last.2.max(b.2);
                last.3 = last.3.max(b.3);
                continue;
            }
        }
        merged.push(b);
    }
    merged
}

fn normalize_detection_boxes(mut boxes: Vec<BoxRect>) -> Vec<BoxRect> {
    dedupe_box_candidates_with_overlap_threshold(
        boxes,
        DETECTION_CONTAINMENT_OVERLAP,
        DETECTION_SIMILARITY_OVERLAP,
    )
}

fn dedupe_box_candidates(mut boxes: Vec<BoxRect>) -> Vec<BoxRect> {
    dedupe_box_candidates_with_overlap_threshold(
        boxes,
        DETECTION_CONTAINMENT_OVERLAP,
        DETECTION_SIMILARITY_OVERLAP,
    )
}

fn dedupe_box_candidates_with_overlap_threshold(
    mut boxes: Vec<BoxRect>,
    containment_overlap: f32,
    similarity_overlap: f32,
) -> Vec<BoxRect> {
    if boxes.len() <= 1 {
        return boxes;
    }

    boxes.retain(|b| b.2 > b.0 && b.3 > b.1);
    if boxes.len() <= 1 {
        return boxes;
    }

    let mut deduped = Vec::with_capacity(boxes.len());
    let mut seen: HashSet<BoxRect> = HashSet::with_capacity(boxes.len());
    for b in boxes {
        if seen.insert(b) {
            deduped.push(b);
        }
    }
    if deduped.len() <= 1 {
        return deduped;
    }
    boxes = deduped;

    boxes.sort_by(|a, b| box_area(*b).cmp(&box_area(*a)));

    let max_bucket = boxes
        .iter()
        .map(|b| b.3)
        .max()
        .unwrap_or(0)
        .saturating_add(BOX_DEDUPE_BUCKET_SIZE.saturating_sub(1))
        .max(1);
    let bucket_count = ((max_bucket as usize) / BOX_DEDUPE_BUCKET_SIZE as usize).saturating_add(1);
    let mut kept: Vec<BoxRect> = Vec::with_capacity(boxes.len());
    let mut bucketed_kept: Vec<Vec<usize>> = vec![Vec::new(); bucket_count.max(1)];

    for candidate in boxes {
        let mut redundant = false;
        let (start_bucket, end_bucket) = box_bucket_range(candidate, bucket_count);
        for bucket_idx in start_bucket..=end_bucket {
            for &kept_idx in &bucketed_kept[bucket_idx] {
                let kept_box = kept[kept_idx];
                if candidate.2 <= kept_box.0
                    || candidate.0 >= kept_box.2
                    || candidate.3 <= kept_box.1
                    || candidate.1 >= kept_box.3
                {
                    continue;
                }
                let overlap = box_intersection_area(candidate, kept_box);
                if overlap == 0 {
                    continue;
                }
                let candidate_area = box_area(candidate).max(1);
                let kept_area = box_area(kept_box).max(1);
                if overlap as f32 / candidate_area.min(kept_area) as f32 >= containment_overlap
                    || box_iou(candidate, kept_box) >= similarity_overlap
                {
                    redundant = true;
                    break;
                }
            }
            if redundant {
                break;
            }
        }
        if redundant {
            continue;
        }

        let kept_idx = kept.len();
        kept.push(candidate);
        for bucket_idx in start_bucket..=end_bucket {
            bucketed_kept[bucket_idx].push(kept_idx);
        }
    }

    kept.sort_by(reading_box_order);
    kept
}

fn box_bucket_range(candidate: BoxRect, bucket_count: usize) -> (usize, usize) {
    let start = (candidate.1 / BOX_DEDUPE_BUCKET_SIZE) as usize;
    let end = (candidate.3.saturating_sub(1) / BOX_DEDUPE_BUCKET_SIZE) as usize;
    (
        start.min(bucket_count.saturating_sub(1)),
        end.min(bucket_count.saturating_sub(1)),
    )
}

fn retain_top_scored_candidates_with_counts(
    mut scored: Vec<(BoxRect, usize, u64)>,
    max_count: usize,
) -> Vec<(BoxRect, usize, u64)> {
    if scored.is_empty() || max_count == 0 {
        return Vec::new();
    }
    let keep = max_count.max(1);
    if scored.len() > keep {
        scored.select_nth_unstable_by(keep - 1, |a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
        scored.truncate(keep);
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    scored
}

fn score_reading_order_cmp(a: &(BoxRect, i64), b: &(BoxRect, i64)) -> std::cmp::Ordering {
    b.1.cmp(&a.1).then_with(|| reading_box_order(&a.0, &b.0))
}

fn retain_top_scored_candidates(
    mut scored: Vec<(BoxRect, i64)>,
    max_count: usize,
) -> Vec<(BoxRect, i64)> {
    if scored.is_empty() || max_count == 0 {
        return Vec::new();
    }
    if scored.len() <= max_count {
        scored.sort_by(score_reading_order_cmp);
        return scored;
    }
    let keep = max_count.max(1);
    scored.select_nth_unstable_by(keep - 1, score_reading_order_cmp);
    scored.truncate(keep);
    scored.sort_by(score_reading_order_cmp);
    scored
}

fn boxes_have_merge_separator(rgb: &image::RgbImage, left: BoxRect, right: BoxRect) -> bool {
    if right.0 <= left.2 || left.3 <= right.1 || right.3 <= left.1 {
        return false;
    }
    let gap = right.0 - left.2;
    let line_h = box_height(left).min(box_height(right)).max(1);
    if gap < ((line_h * 2) / 3).max(10) {
        return false;
    }

    let (w, h) = rgb.dimensions();
    let y0 = left.1.max(right.1).saturating_sub((line_h / 4).max(2));
    let y1 = left
        .3
        .min(right.3)
        .saturating_add((line_h / 4).max(2))
        .min(h);
    let gap_box = clamp_box((left.2.min(w), y0, right.0.min(w), y1), w, h);
    if box_width(gap_box) < gap || box_height(gap_box) < 4 {
        return false;
    }

    gap_region_is_low_texture(rgb, gap_box) || gap_region_has_vertical_separator(rgb, gap_box)
}

fn visual_separator_between_boxes(rgb: &image::RgbImage, a: BoxRect, b: BoxRect) -> bool {
    if a.2 <= b.0 && vertical_overlap(a, b) > 0 {
        return boxes_have_merge_separator(rgb, a, b);
    }
    if b.2 <= a.0 && vertical_overlap(a, b) > 0 {
        return boxes_have_merge_separator(rgb, b, a);
    }
    if a.3 <= b.1 && horizontal_overlap(a, b) > 0 {
        return boxes_have_vertical_merge_separator(rgb, a, b);
    }
    if b.3 <= a.1 && horizontal_overlap(a, b) > 0 {
        return boxes_have_vertical_merge_separator(rgb, b, a);
    }
    false
}

fn boxes_have_vertical_merge_separator(
    rgb: &image::RgbImage,
    upper: BoxRect,
    lower: BoxRect,
) -> bool {
    if lower.1 <= upper.3 || upper.2 <= lower.0 || lower.2 <= upper.0 {
        return false;
    }
    let gap = lower.1 - upper.3;
    let line_h = box_height(upper).min(box_height(lower)).max(1);
    if gap < (line_h / 2).max(12) {
        return false;
    }

    let (w, h) = rgb.dimensions();
    let overlap_x0 = upper.0.max(lower.0);
    let overlap_x1 = upper.2.min(lower.2);
    let x_pad = (box_width(upper).min(box_width(lower)) / 20).clamp(2, 12);
    let gap_box = clamp_box(
        (
            overlap_x0.saturating_sub(x_pad),
            upper.3.min(h),
            overlap_x1.saturating_add(x_pad).min(w),
            lower.1.min(h),
        ),
        w,
        h,
    );
    if box_width(gap_box) < 8 || box_height(gap_box) < gap {
        return false;
    }

    gap_region_is_low_texture(rgb, gap_box) || gap_region_has_horizontal_separator(rgb, gap_box)
}

fn gap_region_is_low_texture(rgb: &image::RgbImage, b: BoxRect) -> bool {
    let area = box_area(b).max(1);
    let mut edge_count = 0u64;
    for y in b.1.saturating_add(1)..b.3.saturating_sub(1) {
        for x in b.0.saturating_add(1)..b.2.saturating_sub(1) {
            let left = rgb.get_pixel(x - 1, y);
            let right = rgb.get_pixel(x + 1, y);
            let up = rgb.get_pixel(x, y - 1);
            let down = rgb.get_pixel(x, y + 1);
            if luma_abs_diff(left, right) >= 14
                || luma_abs_diff(up, down) >= 14
                || color_distance_u8(left, [right[0], right[1], right[2]]) >= 22
                || color_distance_u8(up, [down[0], down[1], down[2]]) >= 22
            {
                edge_count += 1;
            }
        }
    }
    edge_count.saturating_mul(100) <= area.saturating_mul(2)
}

fn gap_region_has_horizontal_separator(rgb: &image::RgbImage, b: BoxRect) -> bool {
    if box_width(b) < 8 || box_height(b) < 3 {
        return false;
    }

    for y in b.1.saturating_add(1)..b.3.saturating_sub(1) {
        let mut edge_count = 0u32;
        for x in b.0.saturating_add(1)..b.2.saturating_sub(1) {
            let up = rgb.get_pixel(x, y - 1);
            let down = rgb.get_pixel(x, y + 1);
            if luma_abs_diff(up, down) >= 18
                || color_distance_u8(up, [down[0], down[1], down[2]]) >= 26
            {
                edge_count += 1;
            }
        }
        if edge_count.saturating_mul(100) >= box_width(b).saturating_mul(72) {
            return true;
        }
    }
    false
}

fn gap_region_has_vertical_separator(rgb: &image::RgbImage, b: BoxRect) -> bool {
    if box_width(b) < 3 || box_height(b) < 8 {
        return false;
    }
    let mut best_col = 0u32;
    for x in b.0.saturating_add(1)..b.2.saturating_sub(1) {
        let mut col_edges = 0u32;
        for y in b.1..b.3 {
            let left = rgb.get_pixel(x - 1, y);
            let right = rgb.get_pixel(x + 1, y);
            if luma_abs_diff(left, right) >= 22
                || color_distance_u8(left, [right[0], right[1], right[2]]) >= 30
            {
                col_edges += 1;
            }
        }
        best_col = best_col.max(col_edges);
    }
    best_col.saturating_mul(100) >= box_height(b).saturating_mul(55)
}

fn raw_split_detection_candidates(merged: BoxRect, raw: &[BoxRect]) -> Vec<BoxRect> {
    if box_width(merged) < 96 {
        return Vec::new();
    }
    let mut contained = raw
        .iter()
        .copied()
        .filter(|b| raw_box_is_split_candidate(*b, merged))
        .take(MAX_RAW_DET_SPLIT_CANDIDATES)
        .collect::<Vec<_>>();
    if contained.len() < 2 {
        return Vec::new();
    }
    contained.sort_by(reading_box_order);
    contained
}

fn raw_box_is_split_candidate(raw: BoxRect, merged: BoxRect) -> bool {
    let raw_area = box_area(raw).max(1);
    let contained = box_intersection_area(raw, merged) as f32 / raw_area as f32;
    if contained < 0.82 {
        return false;
    }
    if box_iou(raw, merged) > 0.92 {
        return false;
    }
    if box_area(raw).saturating_mul(100) >= box_area(merged).saturating_mul(78) {
        return false;
    }
    box_width(raw) + 8 < box_width(merged)
}

fn page_region_boxes(image: &DynamicImage, detection_boxes: &[DetectionBox]) -> Vec<BoxRect> {
    let mut boxes = visual_page_region_boxes(image);
    let det_boxes = page_region_boxes_from_detection_boxes(detection_boxes, image.dimensions());
    if boxes.is_empty() {
        boxes = det_boxes;
    } else {
        for det_box in det_boxes {
            if boxes.len() >= MAX_PAGE_REGION_DET_PASSES {
                break;
            }
            if boxes
                .iter()
                .any(|existing| boxes_significantly_overlap(*existing, det_box))
            {
                continue;
            }
            boxes.push(det_box);
        }
    }
    boxes = sort_and_truncate_by(boxes, MAX_PAGE_REGION_DET_PASSES, |a, b| {
        (a.0, a.1).cmp(&(b.0, b.1))
    });
    boxes
}

fn panel_child_candidate_boxes(image: &DynamicImage) -> Vec<BoxRect> {
    let mut candidates = Vec::<(BoxRect, i64)>::new();
    for b in visual_page_region_boxes(image) {
        let score = supplemental_text_candidate_score(image, b).unwrap_or_default();
        candidates.push((b, score + 12));
    }
    for b in color_region_boxes(image) {
        if box_width(b) < 48 || box_height(b) < 18 {
            continue;
        }
        if box_area(b).saturating_mul(100) > box_area(image_box(image)).saturating_mul(92) {
            continue;
        }
        let score = supplemental_text_candidate_score(image, b).unwrap_or_default();
        candidates.push((b, score as i64));
    }

    if candidates.len() < 2 {
        return Vec::new();
    }
    let candidate_budget = MAX_PANEL_CHILD_CANDIDATES.saturating_mul(4).max(16);
    candidates = retain_top_scored_candidates(candidates, candidate_budget);

    let mut kept = Vec::new();
    for (b, _) in candidates {
        if kept
            .iter()
            .any(|existing: &BoxRect| boxes_significantly_overlap(*existing, b))
        {
            continue;
        }
        kept.push(b);
        if kept.len() >= MAX_PANEL_CHILD_CANDIDATES {
            break;
        }
    }
    kept.sort_by(reading_box_order);
    kept
}

fn should_try_low_threshold_panel_det(image: &DynamicImage, recognized: &RecognizedText) -> bool {
    if recognized.text.trim().is_empty() {
        return panel_text_score(image, image_box(image)) >= 12;
    }
    if recognized.confidence < 0.64 {
        return true;
    }
    if regions_have_repairable_lines(&recognized.regions) {
        return true;
    }
    recognized.line_count <= 1 && panel_text_score(image, image_box(image)) >= 18
}

fn low_threshold_box_thresh(base: f32) -> f32 {
    (base * 0.72).clamp(0.10, 0.18)
}

fn panel_text_score(image: &DynamicImage, b: BoxRect) -> i64 {
    supplemental_text_candidate_score(image, b).unwrap_or_default()
}

fn high_res_tile_boxes(image: &DynamicImage, det_img_side: usize) -> Vec<BoxRect> {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let det_side = det_img_side.max(320) as u32;
    let tile_side = ((det_side * 2) / 3).clamp(384, 720);
    if w <= tile_side && h <= tile_side {
        return Vec::new();
    }

    let overlap = (tile_side / 5).clamp(96, 160);
    let xs = tile_axis_starts(w, tile_side, overlap);
    let ys = tile_axis_starts(h, tile_side, overlap);
    let mut scored = Vec::new();
    for y in ys {
        for x in &xs {
            let x1 = x.saturating_add(tile_side).min(w);
            let y1 = y.saturating_add(tile_side).min(h);
            let b = clamp_box((*x, y, x1, y1), w, h);
            let score = tile_text_score(image, b);
            if score > 0 {
                scored.push((b, score as i64));
            }
        }
    }

    if scored.is_empty() {
        return Vec::new();
    }
    let candidate_budget = MAX_HIGH_RES_TILE_DET_PASSES.saturating_mul(2).max(12);
    scored = retain_top_scored_candidates(scored, candidate_budget);
    scored.sort_by(|a, b| reading_box_order(&a.0, &b.0));
    scored.into_iter().map(|(b, _)| b).collect()
}

fn uncovered_visual_text_boxes(
    image: &DynamicImage,
    existing_regions: &[OcrTextRegion],
) -> Vec<BoxRect> {
    let (img_w, img_h) = image.dimensions();
    if img_w == 0 || img_h == 0 {
        return Vec::new();
    }

    let mut boxes = Vec::new();
    for line_box in foreground_line_boxes(image, 96) {
        let split = split_line_box_horizontally(image, line_box);
        for b in split {
            let b = clamp_box(
                (
                    b.0.saturating_sub(2),
                    b.1.saturating_sub(1),
                    b.2.saturating_add(2),
                    b.3.saturating_add(1),
                ),
                img_w,
                img_h,
            );
            if box_width(b) < 12 || box_height(b) < 6 {
                continue;
            }
            if box_width(b).saturating_mul(100) > img_w.saturating_mul(92) {
                continue;
            }
            if color_region_box_covered_by_reliable_text(b, existing_regions) {
                continue;
            }
            boxes.push(b);
        }
    }

    boxes = nms_boxes(boxes, 0.55);
    boxes = sort_and_truncate_by(
        boxes,
        MAX_EAGER_VISUAL_REGION_RECOGNITIONS.saturating_mul(2),
        reading_box_order,
    );
    boxes
}

fn tile_axis_starts(len: u32, tile_side: u32, overlap: u32) -> Vec<u32> {
    if len <= tile_side {
        return vec![0];
    }
    let stride = tile_side.saturating_sub(overlap).max(1);
    let last = len - tile_side;
    let mut starts = Vec::new();
    let mut cur = 0u32;
    loop {
        if starts.last().copied() != Some(cur) {
            starts.push(cur);
        }
        if cur >= last {
            break;
        }
        let next = cur.saturating_add(stride);
        cur = if next >= last { last } else { next };
    }
    starts
}

fn tile_text_score(image: &DynamicImage, b: BoxRect) -> usize {
    let crop = crop_box(image, b);
    let rgb = to_rgb_on_white(&crop);
    if let Some(edge_mask) = visual_layout_edge_mask_from_rgb(&rgb) {
        let edge_count = edge_mask.iter().filter(|active| **active).count();
        if edge_count > 0 {
            return edge_count;
        }
    }
    text_foreground_mask_from_rgb(&rgb)
        .map(|mask| mask.iter().filter(|active| **active).count())
        .unwrap_or(0)
}

fn visual_page_region_boxes(image: &DynamicImage) -> Vec<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (src_w, src_h) = rgb.dimensions();
    if src_w < 640 || src_h < 240 {
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

    let (w, h) = work.dimensions();
    let Some(mask) = visual_layout_edge_mask_from_rgb(&work) else {
        return Vec::new();
    };
    let column_boxes = visual_page_region_boxes_from_mask(&mask, w as usize, h as usize);
    let grid_boxes = visual_page_grid_region_boxes_from_mask(&mask, w as usize, h as usize);
    let boxes = if grid_boxes.len() > column_boxes.len() {
        grid_boxes
    } else {
        column_boxes
    };
    boxes
        .into_iter()
        .map(|b| {
            let x0 = ((b.0 as f32) * sx).floor().max(0.0) as u32;
            let y0 = ((b.1 as f32) * sy).floor().max(0.0) as u32;
            let x1 = ((b.2 as f32) * sx).ceil().min(src_w as f32) as u32;
            let y1 = ((b.3 as f32) * sy).ceil().min(src_h as f32) as u32;
            clamp_box((x0, y0, x1, y1), src_w, src_h)
        })
        .collect()
}

fn visual_layout_edge_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
    let (w_u32, h_u32) = rgb.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    if w < 8 || h < 8 {
        return None;
    }

    let mut mask = vec![false; w.saturating_mul(h)];
    let mut active = 0usize;
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let left = rgb.get_pixel((x - 1) as u32, y as u32);
            let right = rgb.get_pixel((x + 1) as u32, y as u32);
            let up = rgb.get_pixel(x as u32, (y - 1) as u32);
            let down = rgb.get_pixel(x as u32, (y + 1) as u32);
            let edge = color_distance_u8(left, [right[0], right[1], right[2]]) >= 24
                || color_distance_u8(up, [down[0], down[1], down[2]]) >= 24
                || luma_abs_diff(left, right) >= 18
                || luma_abs_diff(up, down) >= 18;
            if edge {
                mask[y * w + x] = true;
                active += 1;
            }
        }
    }

    let total = w.saturating_mul(h).max(1);
    let ratio = active as f32 / total as f32;
    if active < h.max(32) || !(0.0005..=0.28).contains(&ratio) {
        return None;
    }
    Some(mask)
}

fn luma_abs_diff(a: &image::Rgb<u8>, b: &image::Rgb<u8>) -> u8 {
    luma_u8(a).abs_diff(luma_u8(b))
}

fn luma_u8(pixel: &image::Rgb<u8>) -> u8 {
    ((pixel[0] as u16 * 30 + pixel[1] as u16 * 59 + pixel[2] as u16 * 11) / 100) as u8
}

fn visual_page_region_boxes_from_mask(mask: &[bool], w: usize, h: usize) -> Vec<BoxRect> {
    if w < 80 || h < 80 || mask.len() != w.saturating_mul(h) {
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

    let radius = (w / 320).clamp(2, 8);
    let mut smooth = vec![0usize; w];
    for (x, value) in smooth.iter_mut().enumerate() {
        let x0 = x.saturating_sub(radius);
        let x1 = (x + radius + 1).min(w);
        *value = col_score[x0..x1].iter().sum::<usize>() / (x1 - x0).max(1);
    }

    let max_score = smooth.iter().copied().max().unwrap_or(0);
    if max_score < 3 {
        return Vec::new();
    }
    let active_threshold = ((max_score as f32) * 0.08)
        .ceil()
        .max((h as f32 / 700.0).ceil())
        .max(2.0) as usize;
    let bridge_gap = (w / 24).clamp(40, 110);
    let min_band_width = (w / 18).clamp(80, 220);

    let mut bands: Vec<(usize, usize, u64)> = Vec::new();
    let mut x = 0usize;
    while x < w {
        if smooth[x] < active_threshold {
            x += 1;
            continue;
        }

        let start = x;
        let mut end = x;
        let mut gap = 0usize;
        let mut score = 0u64;
        while x < w {
            if smooth[x] >= active_threshold {
                end = x;
                gap = 0;
                score = score.saturating_add(smooth[x] as u64);
            } else {
                gap += 1;
                if gap > bridge_gap {
                    break;
                }
            }
            x += 1;
        }

        if end.saturating_sub(start) + 1 >= min_band_width && score > 0 {
            bands.push((start, end, score));
        }
    }

    if bands.len() < 2 {
        return Vec::new();
    }

    let mut candidates: Vec<(BoxRect, usize)> = Vec::new();
    for idx in 0..bands.len() {
        let (band_x0, band_x1, score) = bands[idx];
        let x0 = if idx == 0 {
            0
        } else {
            (bands[idx - 1].1.saturating_add(band_x0)) / 2
        };
        let x1 = if idx + 1 == bands.len() {
            w
        } else {
            (band_x1.saturating_add(bands[idx + 1].0)) / 2
        };
        let b = clamp_box((x0 as u32, 0, x1 as u32, h as u32), w as u32, h as u32);
        let region_width = box_width(b);
        if region_width < min_band_width as u32
            || region_width.saturating_mul(100) > (w as u32).saturating_mul(92)
        {
            continue;
        }
        candidates.push((b, score as usize));
    }

    if candidates.len() < 2 {
        return Vec::new();
    }
    let candidate_budget = MAX_PAGE_REGION_DET_PASSES.saturating_mul(4).max(12);
    candidates = retain_top_scored_candidates_with_counts(
        candidates
            .into_iter()
            .map(|(b, score)| (b, 0usize, score as u64))
            .collect(),
        candidate_budget,
    )
    .into_iter()
    .map(|(b, _, _)| (b, 0usize))
    .collect();
    candidates.sort_by_key(|(b, _)| (b.0, b.1));
    candidates.into_iter().map(|(b, _)| b).collect()
}

fn visual_page_grid_region_boxes_from_mask(mask: &[bool], w: usize, h: usize) -> Vec<BoxRect> {
    if w < 160 || h < 160 || mask.len() != w.saturating_mul(h) {
        return Vec::new();
    }

    let row_bands = projection_bands(
        |idx| {
            let mut count = 0usize;
            for x in 0..w {
                if mask[idx * w + x] {
                    count += 1;
                }
            }
            count
        },
        h,
        ((w as f32) * 0.004).ceil().max(3.0) as usize,
        (h / 30).clamp(16, 72),
        (h / 16).clamp(44, 150),
    );
    if row_bands.len() < 2 {
        return Vec::new();
    }

    let mut candidates: Vec<(BoxRect, usize)> = Vec::new();
    for (row_start, row_end) in row_bands {
        let band_h = row_end.saturating_sub(row_start) + 1;
        let col_bands = projection_bands(
            |x| {
                let mut count = 0usize;
                for y in row_start..=row_end {
                    if mask[y * w + x] {
                        count += 1;
                    }
                }
                count
            },
            w,
            ((band_h as f32) * 0.06).ceil().max(2.0) as usize,
            (w / 36).clamp(20, 90),
            (w / 18).clamp(70, 220),
        );
        if col_bands.is_empty() {
            continue;
        }
        for (col_start, col_end) in col_bands {
            let mut score = 0usize;
            for y in row_start..=row_end {
                for x in col_start..=col_end {
                    if mask[y * w + x] {
                        score += 1;
                    }
                }
            }
            if score < 8 {
                continue;
            }
            let pad_x = (w / 160).clamp(6, 18);
            let pad_y = (h / 180).clamp(4, 14);
            let b = clamp_box(
                (
                    col_start.saturating_sub(pad_x) as u32,
                    row_start.saturating_sub(pad_y) as u32,
                    (col_end + 1 + pad_x).min(w) as u32,
                    (row_end + 1 + pad_y).min(h) as u32,
                ),
                w as u32,
                h as u32,
            );
            if box_width(b) < 64 || box_height(b) < 32 {
                continue;
            }
            if box_area(b).saturating_mul(100)
                > (w as u64).saturating_mul(h as u64).saturating_mul(72)
            {
                continue;
            }
            candidates.push((b, score));
        }
    }

    if candidates.len() < 2 {
        return Vec::new();
    }
    let candidate_budget = MAX_PAGE_REGION_DET_PASSES.saturating_mul(4).max(12);
    candidates = retain_top_scored_candidates_with_counts(
        candidates
            .into_iter()
            .map(|(b, score)| (b, 0usize, score as u64))
            .collect(),
        candidate_budget,
    )
    .into_iter()
    .map(|(b, _, _)| (b, 0usize))
    .collect();
    candidates.sort_by(|a, b| reading_box_order(&a.0, &b.0));
    candidates.into_iter().map(|(b, _)| b).collect()
}

fn projection_bands<F>(
    mut score_at: F,
    len: usize,
    active_threshold: usize,
    bridge_gap: usize,
    min_band_len: usize,
) -> Vec<(usize, usize)>
where
    F: FnMut(usize) -> usize,
{
    if len == 0 {
        return Vec::new();
    }
    let mut scores = Vec::with_capacity(len);
    for idx in 0..len {
        scores.push(score_at(idx));
    }
    let max_score = scores.iter().copied().max().unwrap_or(0);
    if max_score < active_threshold {
        return Vec::new();
    }
    let active_threshold = active_threshold.max(((max_score as f32) * 0.06).ceil() as usize);
    let bridge_threshold = (active_threshold / 2).max(1);

    let mut bands = Vec::new();
    let mut idx = 0usize;
    while idx < len {
        if scores[idx] < active_threshold {
            idx += 1;
            continue;
        }
        let start = idx;
        let mut end = idx;
        let mut gap = 0usize;
        idx += 1;
        while idx < len {
            if scores[idx] >= bridge_threshold {
                end = idx;
                gap = 0;
            } else {
                gap += 1;
                if gap > bridge_gap {
                    break;
                }
            }
            idx += 1;
        }
        if end.saturating_sub(start) + 1 >= min_band_len {
            bands.push((start, end));
        }
    }
    bands
}

fn page_region_boxes_from_detection_boxes(
    boxes: &[DetectionBox],
    dimensions: (u32, u32),
) -> Vec<BoxRect> {
    let (img_w, img_h) = dimensions;
    if img_w < 640 || img_h < 320 || boxes.len() < 8 {
        return Vec::new();
    }

    let max_seed_width = (img_w / 4).clamp(180, 520);
    let mut seeds = boxes
        .iter()
        .map(|det_box| det_box.bbox)
        .filter(|b| box_width(*b) >= 16 && box_height(*b) >= 6)
        .filter(|b| box_width(*b) <= max_seed_width)
        .map(|b| {
            let pad = (box_width(b) / 10).clamp(8, 32);
            let x0 = b.0.saturating_sub(pad);
            let x1 = b.2.saturating_add(pad).min(img_w);
            (x0, x1, (x0.saturating_add(x1)) / 2, box_area(b), 1usize)
        })
        .collect::<Vec<_>>();
    if seeds.len() < 4 {
        return Vec::new();
    }

    seeds.sort_by_key(|(x0, x1, cx, _, _)| (*cx, *x0, *x1));
    let center_gap = (img_w / 9).clamp(120, 260);
    let mut bands: Vec<(u32, u32, u32, u64, usize)> = Vec::new();
    for (x0, x1, cx, area, count) in seeds {
        if let Some(last) = bands.last_mut()
            && cx <= last.2.saturating_add(center_gap)
        {
            let total_count = last.4 + count;
            last.1 = last.1.max(x1);
            last.0 = last.0.min(x0);
            last.2 = ((last.2 as u64 * last.4 as u64 + cx as u64 * count as u64)
                / total_count.max(1) as u64) as u32;
            last.3 = last.3.saturating_add(area);
            last.4 = total_count;
            continue;
        }
        bands.push((x0, x1, cx, area, count));
    }

    let min_region_width = (img_w / 18).clamp(80, 180);
    bands.retain(|(x0, x1, _, _, count)| x1.saturating_sub(*x0) >= min_region_width || *count >= 2);
    if bands.len() < 2 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for idx in 0..bands.len() {
        let (band_x0, band_x1, _center, score_area, score_count) = bands[idx];
        let x0 = if idx == 0 {
            0
        } else {
            (bands[idx - 1].1.saturating_add(band_x0)) / 2
        };
        let x1 = if idx + 1 == bands.len() {
            img_w
        } else {
            (band_x1.saturating_add(bands[idx + 1].0)) / 2
        };
        let region = clamp_box((x0, 0, x1, img_h), img_w, img_h);
        let region_width = box_width(region);
        if region_width < min_region_width
            || region_width.saturating_mul(100) > img_w.saturating_mul(92)
        {
            continue;
        }
        candidates.push((region, score_count, score_area));
    }
    if candidates.len() < 2 {
        return Vec::new();
    }

    let candidate_budget = MAX_PAGE_REGION_DET_PASSES.saturating_mul(4).max(12);
    candidates = retain_top_scored_candidates_with_counts(candidates, candidate_budget);
    candidates.sort_by_key(|(b, _, _)| (b.0, b.1));
    candidates.into_iter().map(|(b, _, _)| b).collect()
}

fn reading_box_order(a: &BoxRect, b: &BoxRect) -> std::cmp::Ordering {
    let ya = a.1 as i32 / 8;
    let yb = b.1 as i32 / 8;
    ya.cmp(&yb).then_with(|| a.0.cmp(&b.0))
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

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn reset_ocr_rec_perf() {
    OCR_REC_PERF.with(|perf| {
        *perf.borrow_mut() = OcrRecPerf::default();
    });
    OCR_WORK_PERF.with(|perf| {
        *perf.borrow_mut() = OcrWorkPerf::default();
    });
}

fn record_ocr_rec_perf(variant: RecVariant, elapsed_ms: u64) {
    OCR_REC_PERF.with(|perf| {
        let mut perf = perf.borrow_mut();
        match variant {
            RecVariant::Primary => {
                perf.primary_call_count += 1;
                perf.primary_ms = perf.primary_ms.saturating_add(elapsed_ms);
            }
            RecVariant::Alt => {
                perf.alt_call_count += 1;
                perf.alt_ms = perf.alt_ms.saturating_add(elapsed_ms);
            }
        }
    });
}

fn read_ocr_rec_perf() -> OcrRecPerf {
    OCR_REC_PERF.with(|perf| *perf.borrow())
}

fn ocr_work_perf_record_rec_cache_hit() {
    OCR_WORK_PERF.with(|perf| {
        let mut perf = perf.borrow_mut();
        perf.rec_cache_hit_count = perf.rec_cache_hit_count.saturating_add(1);
    });
}

fn ocr_work_perf_record_rec_cache_miss() {
    OCR_WORK_PERF.with(|perf| {
        let mut perf = perf.borrow_mut();
        perf.rec_cache_miss_count = perf.rec_cache_miss_count.saturating_add(1);
    });
}

fn ocr_work_perf_record_preprocess_call(elapsed_ms: u64) {
    OCR_WORK_PERF.with(|perf| {
        let mut perf = perf.borrow_mut();
        perf.preprocess_call_count = perf.preprocess_call_count.saturating_add(1);
        perf.preprocess_ms = perf.preprocess_ms.saturating_add(elapsed_ms);
    });
}

fn ocr_work_perf_record_variant_candidates(count: usize) {
    OCR_WORK_PERF.with(|perf| {
        let mut perf = perf.borrow_mut();
        perf.variant_candidate_count = perf.variant_candidate_count.saturating_add(count as u64);
    });
}

fn ocr_work_perf_snapshot() -> OcrWorkPerf {
    OCR_WORK_PERF.with(|perf| *perf.borrow())
}

fn dynamic_image_signature_cached(image: &DynamicImage) -> u64 {
    let key = dynamic_image_signature_cache_key(image);
    if let Some(key) = key {
        if let Some(cached) = OCR_REC_IMAGE_SIGNATURE_CACHE.with(|cache| cache.borrow().get(&key)) {
            return cached;
        }
        let signature = dynamic_image_signature_uncached(image);
        OCR_REC_IMAGE_SIGNATURE_CACHE.with(|cache| cache.borrow_mut().put(key, signature));
        signature
    } else {
        dynamic_image_signature_uncached(image)
    }
}

fn dynamic_image_signature(image: &DynamicImage) -> u64 {
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let w = gray.width();
            let h = gray.height();
            let mut hash = ((w as u64) << 32) | h as u64;
            let bytes = gray.as_raw();
            let total = (w as usize).saturating_mul(h as usize).max(1);
            let step = (total / 64).max(1);
            for idx in (0..total).step_by(step) {
                let value = bytes[idx] as u64;
                hash = hash.wrapping_mul(1_099_511_628_211).wrapping_add(value + 1);
            }
            hash
        }
        DynamicImage::ImageRgb8(rgb) => {
            let w = rgb.width();
            let h = rgb.height();
            let mut hash = ((w as u64) << 32) | h as u64;
            let bytes = rgb.as_raw();
            let total = (w as usize).saturating_mul(h as usize).max(1);
            let step = (total / 64).max(1);
            for idx in (0..total).step_by(step) {
                let base = idx * 3;
                if base + 2 >= bytes.len() {
                    break;
                }
                let r = bytes[base] as u64;
                let g = bytes[base + 1] as u64;
                let b = bytes[base + 2] as u64;
                let luma = (r * 77 + g * 150 + b * 29) >> 8;
                hash = hash.wrapping_mul(1_099_511_628_211).wrapping_add(luma + 1);
            }
            hash
        }
        DynamicImage::ImageRgba8(rgba) => {
            let w = rgba.width();
            let h = rgba.height();
            let mut hash = ((w as u64) << 32) | h as u64;
            let bytes = rgba.as_raw();
            let total = (w as usize).saturating_mul(h as usize).max(1);
            let step = (total / 64).max(1);
            for idx in (0..total).step_by(step) {
                let base = idx * 4;
                if base + 2 >= bytes.len() {
                    break;
                }
                let r = bytes[base] as u64;
                let g = bytes[base + 1] as u64;
                let b = bytes[base + 2] as u64;
                let luma = (r * 77 + g * 150 + b * 29) >> 8;
                hash = hash
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add((luma + 1) ^ (bytes[base + 3] as u64));
            }
            hash
        }
        _ => {
            let gray = image.to_luma8();
            let (w, h) = gray.dimensions();
            let mut hash = ((w as u64) << 32) | h as u64;
            let total = (w as usize).saturating_mul(h as usize).max(1);
            let step = (total / 64).max(1);
            for idx in (0..total).step_by(step) {
                hash = hash
                    .wrapping_mul(1_099_511_628_211)
                    .wrapping_add(gray.as_raw()[idx] as u64 + 1);
            }
            hash
        }
    }
}

fn dynamic_image_signature_cache_key(image: &DynamicImage) -> Option<ImageSignatureCacheKey> {
    match image {
        DynamicImage::ImageLuma8(image) => Some(ImageSignatureCacheKey {
            format: 1,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
        }),
        DynamicImage::ImageRgb8(image) => Some(ImageSignatureCacheKey {
            format: 2,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
        }),
        DynamicImage::ImageRgba8(image) => Some(ImageSignatureCacheKey {
            format: 3,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
        }),
        _ => None,
    }
}

fn dynamic_image_signature_uncached(image: &DynamicImage) -> u64 {
    dynamic_image_signature(image)
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

fn preprocess_rec_image_cached(
    image: &DynamicImage,
    target_h: usize,
    target_w: usize,
) -> Result<(Arc<Vec<f32>>, Vec<usize>), String> {
    let image_signature = dynamic_image_signature_cached(image);
    preprocess_rec_image_cached_with_signature(image, image_signature, target_h, target_w)
}

fn preprocess_rec_image_cached_with_signature(
    image: &DynamicImage,
    image_signature: u64,
    target_h: usize,
    target_w: usize,
) -> Result<(Arc<Vec<f32>>, Vec<usize>), String> {
    let key = PreprocessCacheKey {
        image_signature,
        target_w,
        target_h,
    };

    if let Some(cached) = OCR_REC_PREPROCESS_CACHE.with(|cache| cache.borrow_mut().get(&key)) {
        return Ok((cached.input, cached.shape));
    }

    let start = Instant::now();
    let (rec_input, rec_shape) = preprocess_rec_image(image, target_h, target_w)?;
    OCR_REC_PREPROCESS_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .put(key, rec_input.clone(), rec_shape.clone())
    });
    ocr_work_perf_record_preprocess_call(elapsed_ms(start));
    Ok((Arc::new(rec_input), rec_shape))
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

            let bbox_area = max_x
                .saturating_sub(min_x)
                .saturating_add(1)
                .saturating_mul(max_y.saturating_sub(min_y).saturating_add(1))
                .max(1);
            let fill_ratio = positive_area as f32 / bbox_area as f32;
            let refined = contour_refined_boxes_from_component(
                raw_mask, min_x, min_y, max_x, max_y, w, h, min_area,
            );
            if refined.len() >= 2 && (fill_ratio < 0.45 || max_y.saturating_sub(min_y) >= 18) {
                boxes.extend(refined);
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

#[allow(clippy::too_many_arguments)]
fn contour_refined_boxes_from_component(
    raw_mask: &[bool],
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    w: usize,
    h: usize,
    min_area: usize,
) -> Vec<BoxRect> {
    let comp_w = max_x.saturating_sub(min_x).saturating_add(1);
    let comp_h = max_y.saturating_sub(min_y).saturating_add(1);
    if comp_w < 12 || comp_h < 12 || comp_w.saturating_mul(comp_h) < min_area.saturating_mul(2) {
        return Vec::new();
    }

    let mut local = vec![false; comp_w.saturating_mul(comp_h)];
    for y in 0..comp_h {
        for x in 0..comp_w {
            local[y * comp_w + x] = raw_mask[(min_y + y) * w + min_x + x];
        }
    }

    let max_boxes = (comp_h / 6).clamp(2, 8);
    let mut boxes = line_boxes_from_foreground_mask(&local, comp_w, comp_h, max_boxes)
        .into_iter()
        .filter(|b| box_area(*b) as usize >= min_area)
        .map(|b| {
            clamp_box(
                (
                    min_x.saturating_add(b.0 as usize) as u32,
                    min_y.saturating_add(b.1 as usize) as u32,
                    min_x.saturating_add(b.2 as usize) as u32,
                    min_y.saturating_add(b.3 as usize) as u32,
                ),
                w as u32,
                h as u32,
            )
        })
        .collect::<Vec<_>>();
    if boxes.len() < 2 {
        return Vec::new();
    }
    boxes = nms_boxes(boxes, 0.40);
    boxes.sort_by(reading_box_order);
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
    if boxes.is_empty() {
        return Vec::new();
    }
    boxes.sort_by(|a, b| box_area(*b).cmp(&box_area(*a)));

    let bucket_count = boxes
        .iter()
        .map(|b| b.3)
        .max()
        .unwrap_or(0)
        .saturating_add(BOX_DEDUPE_BUCKET_SIZE.saturating_sub(1))
        .max(1);
    let bucket_count = (bucket_count as usize / BOX_DEDUPE_BUCKET_SIZE as usize).saturating_add(1);

    let mut kept: Vec<BoxRect> = Vec::new();
    let mut bucketed_kept: Vec<Vec<usize>> = vec![Vec::new(); bucket_count.max(1)];
    for b in boxes {
        let mut redundant = false;
        let (start_bucket, end_bucket) = box_bucket_range(b, bucketed_kept.len());
        for bucket_idx in start_bucket..=end_bucket {
            for &kept_idx in &bucketed_kept[bucket_idx] {
                if kept_idx >= kept.len() {
                    continue;
                }
                if box_iou(b, kept[kept_idx]) > iou_threshold {
                    redundant = true;
                    break;
                }
            }
            if redundant {
                break;
            }
        }
        if redundant {
            continue;
        }

        let kept_idx = kept.len();
        kept.push(b);
        for bucket_idx in start_bucket..=end_bucket {
            bucketed_kept[bucket_idx].push(kept_idx);
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

fn tight_rec_crop(image: &DynamicImage) -> Option<DynamicImage> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w < 12 || h < 8 {
        return None;
    }
    let mask = text_foreground_mask_from_rgb(&rgb)?;
    let mut min_x = w as usize;
    let mut min_y = h as usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut count = 0usize;
    for y in 0..h as usize {
        for x in 0..w as usize {
            if !mask[y * w as usize + x] {
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

    let fg_h = max_y.saturating_sub(min_y) + 1;
    let pad_x = (fg_h / 2).clamp(2, 12);
    let pad_y = (fg_h / 4).clamp(1, 6);
    let b = (
        min_x.saturating_sub(pad_x) as u32,
        min_y.saturating_sub(pad_y) as u32,
        (max_x + 1 + pad_x).min(w as usize) as u32,
        (max_y + 1 + pad_y).min(h as usize) as u32,
    );
    if box_width(b) < 4 || box_height(b) < 4 {
        return None;
    }
    let image_area = (w as u64).saturating_mul(h as u64);
    if box_area(b).saturating_mul(100) >= image_area.saturating_mul(92) {
        return None;
    }
    Some(crop_box(image, b))
}

fn preprocess_rec_image(
    image: &DynamicImage,
    target_h: usize,
    target_w: usize,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let (src_w, src_h) = image.dimensions();
    if src_w == 0 || src_h == 0 {
        return Err("empty recognition image".to_string());
    }
    if target_h == 0 || target_w == 0 {
        return Err("invalid recognition dimensions".to_string());
    }
    let ratio = src_w as f32 / src_h as f32;
    let mut resized_w = (ratio * target_h as f32).ceil() as usize;
    resized_w = resized_w.clamp(1, target_w);
    let mut data = vec![0f32; 1 * 3 * target_h * target_w];
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let resized = image::imageops::resize(
                gray,
                resized_w as u32,
                target_h as u32,
                FilterType::Triangle,
            );
            let resized_w = resized_w as usize;
            let row_len = target_h * target_w;
            let mid = resized_w.min(target_w);
            let raw = resized.as_raw();
            for y in 0..target_h {
                let row_offset = y * resized_w;
                let out_base = y * target_w;
                for x in 0..mid {
                    let v = (raw[row_offset + x] as f32 - 127.5) / 127.5;
                    let dst = out_base + x;
                    data[dst] = v;
                    data[row_len + dst] = v;
                    data[row_len * 2 + dst] = v;
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            let resized = image::imageops::resize(
                rgb,
                resized_w as u32,
                target_h as u32,
                FilterType::Triangle,
            );
            let resized_w = resized_w as usize;
            let row_len = target_h * target_w;
            let mid = resized_w.min(target_w);
            let raw = resized.as_raw();
            for y in 0..target_h {
                let row_offset = y * resized_w;
                let out_base = y * target_w;
                for x in 0..mid {
                    let src = (row_offset + x) * 3;
                    let dst = out_base + x;
                    data[dst] = (raw[src + 2] as f32 - 127.5) / 127.5;
                    data[row_len + dst] = (raw[src + 1] as f32 - 127.5) / 127.5;
                    data[row_len * 2 + dst] = (raw[src] as f32 - 127.5) / 127.5;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            let resized = image::imageops::resize(
                rgba,
                resized_w as u32,
                target_h as u32,
                FilterType::Triangle,
            );
            let resized_w = resized_w as usize;
            let row_len = target_h * target_w;
            let mid = resized_w.min(target_w);
            let raw = resized.as_raw();
            for y in 0..target_h {
                let row_offset = y * resized_w;
                let out_base = y * target_w;
                for x in 0..mid {
                    let src = (row_offset + x) * 4;
                    let alpha = raw[src + 3] as f32 / 255.0;
                    let dst = out_base + x;
                    if alpha >= 1.0 {
                        data[dst] = (raw[src + 2] as f32 - 127.5) / 127.5;
                        data[row_len + dst] = (raw[src + 1] as f32 - 127.5) / 127.5;
                        data[row_len * 2 + dst] = (raw[src] as f32 - 127.5) / 127.5;
                        continue;
                    }
                    let inv_alpha = 1.0 - alpha;
                    data[dst] =
                        (((raw[src + 2] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                    data[row_len + dst] =
                        (((raw[src + 1] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                    data[row_len * 2 + dst] =
                        (((raw[src] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                }
            }
        }
        _ => {
            let rgba = image.to_rgba8();
            let resized = image::imageops::resize(
                &rgba,
                resized_w as u32,
                target_h as u32,
                FilterType::Triangle,
            );
            let resized_w = resized_w as usize;
            let row_len = target_h * target_w;
            let mid = resized_w.min(target_w);
            let raw = resized.as_raw();
            for y in 0..target_h {
                let row_offset = y * resized_w;
                let out_base = y * target_w;
                for x in 0..mid {
                    let src = (row_offset + x) * 4;
                    let alpha = raw[src + 3] as f32 / 255.0;
                    let dst = out_base + x;
                    if alpha >= 1.0 {
                        data[dst] = (raw[src + 2] as f32 - 127.5) / 127.5;
                        data[row_len + dst] = (raw[src + 1] as f32 - 127.5) / 127.5;
                        data[row_len * 2 + dst] = (raw[src] as f32 - 127.5) / 127.5;
                        continue;
                    }
                    let inv_alpha = 1.0 - alpha;
                    data[dst] =
                        (((raw[src + 2] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                    data[row_len + dst] =
                        (((raw[src + 1] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                    data[row_len * 2 + dst] =
                        (((raw[src] as f32) * alpha + 255.0 * inv_alpha) / 255.0) * 2.0 - 1.0;
                }
            }
        }
    }
    let shape = vec![1, 3, target_h, target_w];
    Ok((data, shape))
}

fn dynamic_rec_target_width(image: &DynamicImage, target_h: usize, base_w: usize) -> usize {
    let (src_w, src_h) = image.dimensions();
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
    to_rgb_on_background_cached(image, background)
        .as_ref()
        .clone()
}

fn to_rgb_on_background_cached(image: &DynamicImage, background: u8) -> Arc<image::RgbImage> {
    let key = match image_rgb_cache_key(image, background) {
        Some(key) => {
            if let Some(image) = OCR_REC_RGB_CACHE.with(|cache| cache.borrow_mut().get(&key)) {
                return image;
            }
            key
        }
        None => {
            return Arc::new(to_rgb_on_background_uncached(image, background));
        }
    };

    let rgb = to_rgb_on_background_uncached(image, background);
    OCR_REC_RGB_CACHE.with(|cache| {
        cache.borrow_mut().put(key, rgb.clone());
    });
    Arc::new(rgb)
}

fn image_rgb_cache_key(image: &DynamicImage, background: u8) -> Option<ImageRgbCacheKey> {
    match image {
        DynamicImage::ImageLuma8(image) => Some(ImageRgbCacheKey {
            format: 1,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        DynamicImage::ImageRgb8(image) => Some(ImageRgbCacheKey {
            format: 2,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        DynamicImage::ImageRgba8(image) => Some(ImageRgbCacheKey {
            format: 3,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        _ => None,
    }
}

fn to_rgb_on_background_uncached(image: &DynamicImage, background: u8) -> image::RgbImage {
    let bg = background as f32;
    match image {
        DynamicImage::ImageLuma8(image) => {
            let mut out = image::RgbImage::new(image.width(), image.height());
            let src = image.as_raw();
            let mut dst = out.as_mut().iter_mut();
            for sample in src.iter() {
                *dst.next().expect("rgb pixel write") = *sample;
                *dst.next().expect("rgb pixel write") = *sample;
                *dst.next().expect("rgb pixel write") = *sample;
            }
            out
        }
        DynamicImage::ImageRgb8(image) => {
            let mut out = image::RgbImage::new(image.width(), image.height());
            if background == 255 {
                return image.clone();
            }
            image.pixels().zip(out.pixels_mut()).for_each(|(src, dst)| {
                dst.0 = [src.0[0], src.0[1], src.0[2]];
            });
            out
        }
        DynamicImage::ImageRgba8(image) => {
            let rgba = image.as_raw();
            let mut out = Vec::with_capacity(image.width() as usize * image.height() as usize * 3);
            for rgba in rgba.chunks_exact(4) {
                let alpha = rgba[3] as f32 / 255.0;
                let inv_alpha = 1.0 - alpha;
                out.extend_from_slice(&[
                    ((rgba[0] as f32 * alpha + bg * inv_alpha).round() as u8),
                    ((rgba[1] as f32 * alpha + bg * inv_alpha).round() as u8),
                    ((rgba[2] as f32 * alpha + bg * inv_alpha).round() as u8),
                ]);
            }
            image::RgbImage::from_raw(image.width(), image.height(), out)
                .expect("rgb buffer has expected size")
        }
        _ => {
            let rgba = image.to_rgba8();
            to_rgb_on_background_uncached(&DynamicImage::ImageRgba8(rgba), background)
        }
    }
}

fn to_luma_on_white(image: &DynamicImage) -> GrayImage {
    to_luma_on_background(image, 255)
}

fn to_luma_on_background(image: &DynamicImage, background: u8) -> GrayImage {
    to_luma_on_background_cached(image, background)
        .as_ref()
        .clone()
}

fn to_luma_on_background_cached(image: &DynamicImage, background: u8) -> Arc<GrayImage> {
    let key = match image_luma_cache_key(image, background) {
        Some(key) => {
            if let Some(image) = OCR_REC_LUMA_CACHE.with(|cache| cache.borrow_mut().get(&key)) {
                return image;
            }
            key
        }
        None => {
            return Arc::new(to_luma_on_background_uncached(image, background));
        }
    };

    let luma = to_luma_on_background_uncached(image, background);
    OCR_REC_LUMA_CACHE.with(|cache| {
        cache.borrow_mut().put(key, luma.clone());
    });
    Arc::new(luma)
}

fn image_luma_cache_key(image: &DynamicImage, background: u8) -> Option<ImageLumaCacheKey> {
    match image {
        DynamicImage::ImageLuma8(image) => Some(ImageLumaCacheKey {
            format: 1,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        DynamicImage::ImageRgb8(image) => Some(ImageLumaCacheKey {
            format: 2,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        DynamicImage::ImageRgba8(image) => Some(ImageLumaCacheKey {
            format: 3,
            width: image.width(),
            height: image.height(),
            ptr: image.as_ptr() as usize,
            len: image.as_raw().len(),
            background,
        }),
        _ => None,
    }
}

fn to_luma_on_background_uncached(image: &DynamicImage, background: u8) -> GrayImage {
    match image {
        DynamicImage::ImageLuma8(image) => image.clone(),
        _ => to_gray_from_rgb(&to_rgb_on_background_cached(image, background)),
    }
}

fn to_gray_from_rgb(rgb: &image::RgbImage) -> GrayImage {
    let mut out = GrayImage::new(rgb.width(), rgb.height());
    let raw = rgb.as_raw();
    let dst = out.as_mut();
    for i in 0..(rgb.width() as usize).saturating_mul(rgb.height() as usize) {
        let base = i * 3;
        let value =
            ((raw[base] as u16 * 77 + raw[base + 1] as u16 * 150 + raw[base + 2] as u16 * 29) >> 8)
                as u8;
        dst[i] = value;
    }
    out
}

fn has_non_opaque_alpha(image: &DynamicImage) -> bool {
    match image {
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_) => false,
        DynamicImage::ImageRgba8(image) => image.pixels().any(|p| p[3] < 255),
        _ => image.to_rgba8().pixels().any(|p| p[3] < 255),
    }
}

fn ocr_trace_enabled() -> bool {
    std::env::var("VECTRAPARSE_OCR_TRACE").ok().as_deref() == Some("1")
}

fn ocr_trace_json_enabled() -> bool {
    std::env::var("VECTRAPARSE_OCR_TRACE_JSON").ok().as_deref() == Some("1")
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
    enhancement_variants_limited(image, usize::MAX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnhancementBaseVariant {
    Luma,
    Hsl,
    Max,
    AlphaBlackLuma,
    AlphaBlackHsl,
    AlphaBlackMax,
}

#[derive(Debug, Clone, Copy)]
struct EnhancementBaseScore {
    base: EnhancementBaseVariant,
    score: i32,
    order: usize,
}

#[derive(Debug, Clone, Copy)]
struct EnhancementRoi {
    width: u32,
    height: u32,
    area: u64,
    mean: f32,
    contrast: f32,
    foreground_ratio: f32,
    has_alpha: bool,
}

#[derive(Debug, Clone, Copy)]
struct LumaSignal {
    mean: f32,
    contrast: f32,
    width: u32,
    height: u32,
}

fn luma_signal(gray: &GrayImage) -> LumaSignal {
    let w = gray.width();
    let h = gray.height();
    if w == 0 || h == 0 {
        return LumaSignal {
            mean: 127.0,
            contrast: 64.0,
            width: w,
            height: h,
        };
    }

    let mut min_v = u8::MAX;
    let mut max_v = u8::MIN;
    let mut sum = 0u64;
    for px in gray.pixels() {
        let v = px[0];
        min_v = min_v.min(v);
        max_v = max_v.max(v);
        sum += v as u64;
    }
    let total = (w as u64) * (h as u64);
    let mean = sum as f32 / total as f32;

    LumaSignal {
        mean,
        contrast: (max_v as f32 - min_v as f32).max(16.0),
        width: w,
        height: h,
    }
}

fn enhancement_roi_from_gray(
    gray: &GrayImage,
    signal: &LumaSignal,
    has_alpha: bool,
) -> EnhancementRoi {
    let area = (signal.width as u64) * (signal.height as u64);
    let mut foreground = 0u64;
    let total = (signal.width as u64) * (signal.height as u64);
    if total > 0 {
        for px in gray.pixels() {
            let delta = (px[0] as f32 - signal.mean).abs();
            if delta >= 16.0 {
                foreground += 1;
            }
        }
    }
    let foreground_ratio = if total > 0 {
        foreground as f32 / total as f32
    } else {
        0.0
    };
    EnhancementRoi {
        width: signal.width,
        height: signal.height,
        area,
        mean: signal.mean,
        contrast: signal.contrast,
        foreground_ratio,
        has_alpha,
    }
}

fn enhancement_variant_base_score(
    base: EnhancementBaseVariant,
    signal: &LumaSignal,
    roi: &EnhancementRoi,
) -> i32 {
    let area = (signal.width as u64) * (signal.height as u64);
    match base {
        EnhancementBaseVariant::Luma => {
            let mut score = 18;
            if signal.contrast < 28.0 {
                score += 12;
            }
            if signal.mean >= 130.0 && signal.mean <= 185.0 {
                score += 4;
            }
            if area <= 1_600 {
                score += 6;
            }
            if roi.foreground_ratio < 0.003 {
                score += 4;
            }
            score
        }
        EnhancementBaseVariant::Hsl => {
            let mut score = 10;
            if signal.contrast >= 40.0 {
                score += 6;
            }
            if signal.mean < 85.0 || signal.mean > 215.0 {
                score += 8;
            }
            if area <= 1_600 {
                score -= 2;
            }
            if roi.foreground_ratio < 0.004 {
                score -= 2;
            }
            score
        }
        EnhancementBaseVariant::Max => {
            let mut score = 6;
            if signal.contrast >= 52.0 {
                score += 10;
            }
            if signal.mean > 145.0 {
                score -= 4;
            }
            if signal.width <= 72 || signal.height <= 16 {
                score -= 12;
            }
            if roi.contrast < 22.0 {
                score -= 10;
            }
            score
        }
        EnhancementBaseVariant::AlphaBlackLuma
        | EnhancementBaseVariant::AlphaBlackHsl
        | EnhancementBaseVariant::AlphaBlackMax => {
            let mut score = 16;
            if signal.contrast < 36.0 {
                score += 12;
            }
            if signal.mean > 190.0 {
                score -= 8;
            }
            if signal.mean < 60.0 {
                score += 8;
            }
            if !roi.has_alpha {
                score -= 999;
            }
            score
        }
    }
}

fn ranked_enhancement_bases(
    image: &DynamicImage,
    variant_budget: usize,
) -> Vec<EnhancementBaseVariant> {
    if variant_budget <= 2 {
        let gray = to_luma_on_white(image);
        let signal = luma_signal(&gray);
        let mut first = if signal.mean < 90.0 || signal.mean > 190.0 {
            EnhancementBaseVariant::Luma
        } else {
            EnhancementBaseVariant::Luma
        };
        let mut second = if signal.mean < 90.0 {
            EnhancementBaseVariant::Hsl
        } else {
            EnhancementBaseVariant::Luma
        };
        if has_non_opaque_alpha(image) {
            first = if signal.mean < 70.0 {
                EnhancementBaseVariant::AlphaBlackLuma
            } else {
                EnhancementBaseVariant::Luma
            };
            second = EnhancementBaseVariant::Luma;
        }
        return vec![first, second];
    }

    let gray = to_luma_on_white(image);
    let signal = luma_signal(&gray);
    let roi = enhancement_roi_from_gray(&gray, &signal, has_non_opaque_alpha(image));
    let mut scored = Vec::new();
    let mut order = 0usize;

    scored.push(EnhancementBaseScore {
        base: EnhancementBaseVariant::Luma,
        score: enhancement_variant_base_score(EnhancementBaseVariant::Luma, &signal, &roi),
        order,
    });
    order += 1;
    scored.push(EnhancementBaseScore {
        base: EnhancementBaseVariant::Hsl,
        score: enhancement_variant_base_score(EnhancementBaseVariant::Hsl, &signal, &roi),
        order,
    });
    order += 1;
    scored.push(EnhancementBaseScore {
        base: EnhancementBaseVariant::Max,
        score: enhancement_variant_base_score(EnhancementBaseVariant::Max, &signal, &roi),
        order,
    });

    if has_non_opaque_alpha(image) {
        scored.push(EnhancementBaseScore {
            base: EnhancementBaseVariant::AlphaBlackLuma,
            score: enhancement_variant_base_score(
                EnhancementBaseVariant::AlphaBlackLuma,
                &signal,
                &roi,
            ),
            order: order + 3,
        });
        scored.push(EnhancementBaseScore {
            base: EnhancementBaseVariant::AlphaBlackHsl,
            score: enhancement_variant_base_score(
                EnhancementBaseVariant::AlphaBlackHsl,
                &signal,
                &roi,
            ),
            order: order + 4,
        });
        scored.push(EnhancementBaseScore {
            base: EnhancementBaseVariant::AlphaBlackMax,
            score: enhancement_variant_base_score(
                EnhancementBaseVariant::AlphaBlackMax,
                &signal,
                &roi,
            ),
            order: order + 5,
        });
    }

    scored.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.order.cmp(&b.order)));
    if scored.is_empty() {
        vec![EnhancementBaseVariant::Luma]
    } else {
        scored.into_iter().map(|entry| entry.base).collect()
    }
}

fn enhancement_variant_base_image(image: &DynamicImage, base: EnhancementBaseVariant) -> GrayImage {
    match base {
        EnhancementBaseVariant::Luma => to_luma_on_white(image),
        EnhancementBaseVariant::Hsl => to_hsl_lightness(image),
        EnhancementBaseVariant::Max => to_max_channel_gray(image),
        EnhancementBaseVariant::AlphaBlackLuma => to_luma_on_background(image, 0),
        EnhancementBaseVariant::AlphaBlackHsl => to_hsl_lightness_on_background(image, 0),
        EnhancementBaseVariant::AlphaBlackMax => to_max_channel_gray_on_background(image, 0),
    }
}

fn enhancement_variant_prefix(base: EnhancementBaseVariant) -> &'static str {
    match base {
        EnhancementBaseVariant::Luma => "",
        EnhancementBaseVariant::Hsl => "hsl-",
        EnhancementBaseVariant::Max => "max-",
        EnhancementBaseVariant::AlphaBlackLuma => "alpha-black-",
        EnhancementBaseVariant::AlphaBlackHsl => "alpha-black-hsl-",
        EnhancementBaseVariant::AlphaBlackMax => "alpha-black-max-",
    }
}

fn enhancement_variants_limited(
    image: &DynamicImage,
    variant_budget: usize,
) -> Vec<(String, DynamicImage)> {
    let has_alpha = has_non_opaque_alpha(image);
    let full_capacity = if has_alpha { 30 } else { 18 };
    let variant_budget = variant_budget.min(full_capacity);
    if variant_budget == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(variant_budget);
    let bases = if variant_budget >= full_capacity {
        let mut bases = vec![
            EnhancementBaseVariant::Luma,
            EnhancementBaseVariant::Hsl,
            EnhancementBaseVariant::Max,
        ];
        if has_alpha {
            bases.extend([
                EnhancementBaseVariant::AlphaBlackLuma,
                EnhancementBaseVariant::AlphaBlackHsl,
                EnhancementBaseVariant::AlphaBlackMax,
            ]);
        }
        bases
    } else {
        ranked_enhancement_bases(image, variant_budget)
    };

    let gray = to_luma_on_white(image);
    let signal = luma_signal(&gray);
    let roi = enhancement_roi_from_gray(&gray, &signal, has_alpha);
    for base in bases {
        if out.len() >= variant_budget {
            break;
        }
        let gray = enhancement_variant_base_image(image, base);
        let prefix = enhancement_variant_prefix(base);
        push_enhancement_variants_limited(&mut out, prefix, &gray, variant_budget, &roi);
        if out.len() >= variant_budget {
            break;
        }
        if out.len() > 0 && base == EnhancementBaseVariant::Luma && variant_budget <= 3 {
            break;
        }
    }

    out.truncate(variant_budget);
    out
}

fn push_enhancement_variants_limited(
    out: &mut Vec<(String, DynamicImage)>,
    prefix: &str,
    base: &GrayImage,
    variant_budget: usize,
    roi: &EnhancementRoi,
) {
    if out.len() >= variant_budget {
        return;
    }
    let stretched = contrast_stretch_luma(base);
    out.push((
        format!("{prefix}contrast"),
        DynamicImage::ImageLuma8(stretched.clone()),
    ));
    if out.len() >= variant_budget {
        return;
    }

    let include_binary = roi.contrast >= 12.0;
    if !include_binary {
        return;
    }

    let binary = adaptive_binary_luma(&stretched, false);
    out.push((format!("{prefix}binary"), DynamicImage::ImageLuma8(binary)));
    if out.len() >= variant_budget {
        return;
    }

    let include_binary_invert = variant_budget > 1;
    if include_binary_invert {
        let binary_invert = adaptive_binary_luma(&stretched, true);
        out.push((
            format!("{prefix}binary-invert"),
            DynamicImage::ImageLuma8(binary_invert),
        ));
    }
    if out.len() >= variant_budget {
        return;
    }

    let include_local = roi.area >= 5_000 && roi.contrast >= 18.0 || variant_budget >= 6;
    if !include_local {
        return;
    }
    let local_binary = local_binary_luma(&stretched, false);
    out.push((
        format!("{prefix}local-binary"),
        DynamicImage::ImageLuma8(local_binary),
    ));
    if out.len() >= variant_budget {
        return;
    }

    let include_local_invert = variant_budget > 2;
    if include_local_invert {
        let local_binary_invert = local_binary_luma(&stretched, true);
        out.push((
            format!("{prefix}local-binary-invert"),
            DynamicImage::ImageLuma8(local_binary_invert),
        ));
    }
}

fn local_recognition_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let mut out = Vec::with_capacity(12);
    out.extend(foreground_binary_variants(image));

    let gray = to_luma_on_white(image);
    let stretched = contrast_stretch_luma(&gray);
    for (name, radius, bias) in [
        ("local-small", 6usize, 6i16),
        ("local-medium", 12usize, 8i16),
        ("local-large", 20usize, 8i16),
    ] {
        out.push((
            name.to_string(),
            DynamicImage::ImageLuma8(local_binary_luma_with_radius(
                &stretched, false, radius, bias,
            )),
        ));
        out.push((
            format!("{name}-invert"),
            DynamicImage::ImageLuma8(local_binary_luma_with_radius(
                &stretched, true, radius, bias,
            )),
        ));
    }

    out
}

fn local_recognition_variants_limited(
    image: &DynamicImage,
    max_count: usize,
) -> Vec<(String, DynamicImage)> {
    if max_count == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(max_count);
    out.extend(foreground_binary_variants_limited(image, max_count));
    if out.len() >= max_count {
        return dedupe_named_dynamic_variants(out);
    }

    let gray = to_luma_on_white(image);
    let stretched = contrast_stretch_luma(&gray);
    for (name, radius, bias) in [
        ("local-small", 6usize, 6i16),
        ("local-medium", 12usize, 8i16),
        ("local-large", 20usize, 8i16),
    ] {
        if out.len() >= max_count {
            break;
        }
        out.push((
            name.to_string(),
            DynamicImage::ImageLuma8(local_binary_luma_with_radius(
                &stretched, false, radius, bias,
            )),
        ));
        if out.len() >= max_count {
            break;
        }
        out.push((
            format!("{name}-invert"),
            DynamicImage::ImageLuma8(local_binary_luma_with_radius(
                &stretched, true, radius, bias,
            )),
        ));
    }

    out.truncate(max_count);
    dedupe_named_dynamic_variants(out)
}

fn foreground_binary_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    let mut out = Vec::new();
    if let Some(binary) = binarize_color_region_foreground(image, image_box(image)) {
        out.push(("foreground-binary".to_string(), binary));
    }

    for (name, mask) in [
        ("foreground-mask", foreground_mask_from_rgb(&rgb)),
        (
            "foreground-soft-mask",
            soft_color_foreground_mask_from_rgb(&rgb),
        ),
        (
            "foreground-low-contrast-mask",
            low_contrast_foreground_mask_from_rgb(&rgb),
        ),
        ("foreground-dark-mask", dark_luma_mask_from_rgb(&rgb)),
    ] {
        let Some(mask) = mask else {
            continue;
        };
        if !foreground_glyph_textness_score(&mask, w as usize, h as usize)
            .is_some_and(|score| score >= -20)
        {
            continue;
        }
        if let Some(binary) = dynamic_image_from_foreground_mask(&mask, w as usize, h as usize) {
            out.push((name.to_string(), binary));
        }
    }

    dedupe_named_dynamic_variants(out)
}

fn foreground_binary_variants_limited(
    image: &DynamicImage,
    max_count: usize,
) -> Vec<(String, DynamicImage)> {
    if max_count == 0 {
        return Vec::new();
    }

    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    let mut out = Vec::with_capacity(max_count.min(4));
    if let Some(binary) = binarize_color_region_foreground(image, image_box(image)) {
        out.push(("foreground-binary".to_string(), binary));
        if out.len() >= max_count {
            return out;
        }
    }

    for (name, mask) in [
        ("foreground-mask", foreground_mask_from_rgb(&rgb)),
        (
            "foreground-soft-mask",
            soft_color_foreground_mask_from_rgb(&rgb),
        ),
        (
            "foreground-low-contrast-mask",
            low_contrast_foreground_mask_from_rgb(&rgb),
        ),
        ("foreground-dark-mask", dark_luma_mask_from_rgb(&rgb)),
    ] {
        if out.len() >= max_count {
            break;
        }
        let Some(mask) = mask else {
            continue;
        };
        if !foreground_glyph_textness_score(&mask, w as usize, h as usize)
            .is_some_and(|score| score >= -20)
        {
            continue;
        }
        if out.len() >= max_count {
            break;
        }
        if let Some(binary) = dynamic_image_from_foreground_mask(&mask, w as usize, h as usize) {
            out.push((name.to_string(), binary));
        }
    }

    dedupe_named_dynamic_variants(out)
}

fn dynamic_image_from_foreground_mask(mask: &[bool], w: usize, h: usize) -> Option<DynamicImage> {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return None;
    }
    let mut foreground = 0usize;
    let mut out = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let active = mask[y * w + x];
            if active {
                foreground += 1;
            }
            out.put_pixel(x as u32, y as u32, Luma([if active { 0 } else { 255 }]));
        }
    }
    let total = w.saturating_mul(h).max(1);
    let ratio = foreground as f32 / total as f32;
    if foreground < 4 || !(0.001..=0.72).contains(&ratio) {
        return None;
    }
    Some(DynamicImage::ImageLuma8(out))
}

fn dedupe_named_dynamic_variants(
    variants: Vec<(String, DynamicImage)>,
) -> Vec<(String, DynamicImage)> {
    let mut kept = Vec::new();
    let mut seen = HashSet::<u64>::new();
    for (name, image) in variants {
        let key = dynamic_image_signature_cached(&image);
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        kept.push((name, image));
    }
    kept
}

fn local_recognition_variants_adaptive(
    image: &DynamicImage,
    direct: Option<&RecCandidate>,
) -> Vec<(String, DynamicImage)> {
    let mut variant_budget = local_recognition_variant_budget(image, direct);
    if let Some(candidate) = direct
        && candidate.confidence >= MIN_STRONG_REC_CONFIDENCE
        && candidate.avg_margin >= 0.04
        && readable_ratio(&candidate.text) >= 0.65
    {
        variant_budget = variant_budget.min(3);
    }
    local_recognition_variants_limited(image, variant_budget)
}

fn local_recognition_variant_budget(image: &DynamicImage, direct: Option<&RecCandidate>) -> usize {
    let (w, h) = image.dimensions();
    let area = (w as u64).saturating_mul(h as u64);

    if let Some(candidate) = direct {
        let text = normalize_recognized_text(&candidate.text);
        let text_len = text.chars().count();

        if is_stable_short_text_candidate(candidate, &text, MAX_FAST_TEXT_CHARS) {
            return 1;
        }

        let mut budget = if candidate.confidence >= MIN_STRONG_REC_CONFIDENCE
            && candidate.avg_margin >= 0.05
        {
            if text_len >= 12 { 3 } else { 4 }
        } else if candidate.confidence >= 0.78 || (candidate.avg_margin >= 0.10 && text_len >= 8) {
            4
        } else if candidate.confidence < 0.45 && text_len < 6 {
            8
        } else {
            7
        };

        if area <= LOCAL_RECOGNITION_TINY_CROP_AREA {
            if text_len <= 4 {
                return 1;
            }
            budget = budget.min(2);
        }
        if area <= LOCAL_RECOGNITION_SMALL_CROP_AREA && text_len <= 4 {
            return 1;
        }
        if area <= LOCAL_RECOGNITION_SMALL_CROP_AREA && text_len <= 6 {
            budget = budget.min(2);
        }

        return budget;
    }

    if area <= LOCAL_RECOGNITION_TINY_CROP_AREA {
        return 2;
    }
    if area <= LOCAL_RECOGNITION_SMALL_CROP_AREA {
        return 5;
    }

    7
}

fn should_try_local_recognition_variants(
    image: &DynamicImage,
    direct: Option<&RecCandidate>,
) -> bool {
    let Some(candidate) = direct else {
        return true;
    };
    let text = normalize_recognized_text(&candidate.text);
    if is_stable_short_text_candidate(candidate, &text, MAX_FAST_TEXT_CHARS.saturating_sub(2)) {
        return false;
    }
    let text = normalize_recognized_text(&candidate.text);
    if recognized_box_needs_repair(image_box(image), &text, candidate.confidence) {
        return true;
    }
    if candidate.confidence < 0.78 {
        return true;
    }
    if candidate.avg_margin < 0.08 || candidate.min_margin < 0.03 {
        return true;
    }
    false
}

fn should_try_crop_enhancement_variants(direct: Option<&RecCandidate>) -> bool {
    let Some(candidate) = direct else {
        return true;
    };
    if !is_usable_recognition(candidate) {
        return true;
    }
    let text = normalize_recognized_text(&candidate.text);
    if is_stable_short_text_candidate(candidate, &text, MAX_FAST_TEXT_CHARS.saturating_sub(2)) {
        return false;
    }
    if candidate.confidence < 0.78 {
        return true;
    }
    if candidate.avg_margin < 0.08 || candidate.min_margin < 0.03 {
        return true;
    }
    let readable_ratio = readable_ratio(&text);
    let text_len = text.chars().count();
    if candidate.confidence >= 0.90
        && candidate.avg_margin >= 0.12
        && readable_ratio >= 0.82
        && text_len <= 6
    {
        return false;
    }
    if readable_ratio < 0.65 {
        return true;
    }
    if dominant_char_ratio(&text) >= 0.60 || punctuation_ratio(&text) > 0.55 {
        return true;
    }
    false
}

#[inline]
fn crop_enhancement_candidate_is_final(candidate: &RecCandidate) -> bool {
    if !is_usable_recognition(candidate) || candidate.confidence < 0.96 {
        return false;
    }
    if candidate.avg_margin < 0.14 || candidate.min_margin < 0.05 {
        return false;
    }
    let text = normalize_recognized_text(&candidate.text);
    if text.chars().count() <= 1 {
        return false;
    }
    readable_ratio(&text) >= 0.90 && candidate.char_min_confidence >= 0.95
}

fn crop_enhancement_variant_budget(image: &DynamicImage, direct: Option<&RecCandidate>) -> usize {
    if !should_try_crop_enhancement_variants(direct) {
        return 0;
    }

    let (w, h) = image.dimensions();
    let area = (w as u64).saturating_mul(h as u64);
    let mut budget = match area {
        area if area <= 5_000 => 1,
        area if area <= 22_000 => 2,
        area if area <= 96_000 => 3,
        _ => MAX_ENHANCEMENT_VARIANTS_PER_PASS,
    };

    if let Some(candidate) = direct {
        let normalized = normalize_recognized_text(&candidate.text);
        let direct_readable_ratio = readable_ratio(&normalized);
        let text_len = normalized.chars().count();
        if area <= CROP_ENHANCE_TINY_AREA && text_len <= 6 {
            return 1;
        }
        if area <= CROP_ENHANCE_SMALL_AREA && text_len <= 4 {
            return 1;
        }
        if area <= CROP_ENHANCE_SMALL_AREA && text_len <= 8 {
            budget = budget.min(2);
        }
        if is_stable_short_text_candidate(
            candidate,
            &normalized,
            MAX_FAST_TEXT_CHARS.saturating_sub(2),
        ) {
            budget = budget.min(1);
        }
        let quality_visible = candidate.confidence >= 0.86
            && candidate.avg_margin >= 0.10
            && candidate.min_margin >= 0.03
            && direct_readable_ratio >= 0.75;
        let long_text = text_len >= 18;
        if quality_visible && !long_text {
            budget = budget.min(2);
        }
        if area > 22_000 {
            if candidate.confidence < 0.78
                || candidate.avg_margin < 0.08
                || candidate.min_margin < 0.03
            {
                budget = MAX_ENHANCEMENT_VARIANTS_PER_PASS;
            } else if candidate.confidence < 0.86
                || candidate.avg_margin < 0.10
                || direct_readable_ratio < 0.75
            {
                budget = budget.max(3);
            }
        }
    }

    budget
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

fn local_det_upscale_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    if w >= 900 || h >= 900 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (name, scale) in [("2x", 2.0f32), ("1.5x", 1.5f32)] {
        let target_w = ((w as f32) * scale).round() as u32;
        let target_h = ((h as f32) * scale).round() as u32;
        let pixels = (target_w as u64).saturating_mul(target_h as u64);
        if pixels > MAX_UPSCALE_PIXELS {
            continue;
        }
        if target_w <= w || target_h <= h {
            continue;
        }
        let resized = image::imageops::resize(image, target_w, target_h, FilterType::CatmullRom);
        out.push((
            format!("local-upscaled-{name}"),
            DynamicImage::ImageRgba8(resized),
        ));
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

fn deskew_variants(image: &DynamicImage) -> Vec<(String, DynamicImage)> {
    let Some(angle) = estimate_foreground_skew_degrees(image) else {
        return Vec::new();
    };
    if !(1.0..=7.0).contains(&angle.abs()) {
        return Vec::new();
    }
    let corrected = rotate_image_degrees_on_white(image, -angle);
    vec![(format!("{:.1}", -angle), corrected)]
}

fn estimate_foreground_skew_degrees(image: &DynamicImage) -> Option<f32> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w < 24 || h < 12 {
        return None;
    }
    let mask = text_foreground_mask_from_rgb(&rgb)?;
    let mut count = 0usize;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for y in 0..h as usize {
        for x in 0..w as usize {
            if !mask[y * w as usize + x] {
                continue;
            }
            count += 1;
            sum_x += x as f64;
            sum_y += y as f64;
        }
    }
    if count < 16 {
        return None;
    }

    let mean_x = sum_x / count as f64;
    let mean_y = sum_y / count as f64;
    let mut cov_xx = 0.0f64;
    let mut cov_xy = 0.0f64;
    for y in 0..h as usize {
        for x in 0..w as usize {
            if !mask[y * w as usize + x] {
                continue;
            }
            let dx = x as f64 - mean_x;
            let dy = y as f64 - mean_y;
            cov_xx += dx * dx;
            cov_xy += dx * dy;
        }
    }
    if cov_xx <= 1.0 {
        return None;
    }

    let angle = (cov_xy / cov_xx).atan().to_degrees() as f32;
    if angle.is_finite() { Some(angle) } else { None }
}

fn rotate_image_degrees_on_white(image: &DynamicImage, degrees: f32) -> DynamicImage {
    let src = to_rgb_on_white(image);
    let (w, h) = src.dimensions();
    if w == 0 || h == 0 {
        return DynamicImage::ImageRgb8(src);
    }

    let radians = degrees.to_radians();
    let cos = radians.cos();
    let sin = radians.sin();
    let corners = [
        (-(w as f32) / 2.0, -(h as f32) / 2.0),
        ((w as f32) / 2.0, -(h as f32) / 2.0),
        (-(w as f32) / 2.0, (h as f32) / 2.0),
        ((w as f32) / 2.0, (h as f32) / 2.0),
    ];
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        let rx = x * cos - y * sin;
        let ry = x * sin + y * cos;
        min_x = min_x.min(rx);
        max_x = max_x.max(rx);
        min_y = min_y.min(ry);
        max_y = max_y.max(ry);
    }
    let out_w = (max_x - min_x).ceil().max(1.0) as u32;
    let out_h = (max_y - min_y).ceil().max(1.0) as u32;
    let mut out = image::RgbImage::from_pixel(out_w, out_h, image::Rgb([255, 255, 255]));
    let src_cx = (w as f32 - 1.0) / 2.0;
    let src_cy = (h as f32 - 1.0) / 2.0;
    let out_cx = (out_w as f32 - 1.0) / 2.0;
    let out_cy = (out_h as f32 - 1.0) / 2.0;

    for y in 0..out_h {
        for x in 0..out_w {
            let dx = x as f32 - out_cx;
            let dy = y as f32 - out_cy;
            let src_x = dx * cos + dy * sin + src_cx;
            let src_y = -dx * sin + dy * cos + src_cy;
            if src_x >= 0.0 && src_y >= 0.0 && src_x < w as f32 && src_y < h as f32 {
                let px = src.get_pixel(
                    (src_x.round() as u32).min(w.saturating_sub(1)),
                    (src_y.round() as u32).min(h.saturating_sub(1)),
                );
                out.put_pixel(x, y, *px);
            }
        }
    }

    DynamicImage::ImageRgb8(out)
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
    local_binary_luma_with_radius(gray, invert, 12, 8)
}

fn local_binary_luma_with_radius(
    gray: &GrayImage,
    invert: bool,
    radius: usize,
    bias: i16,
) -> GrayImage {
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
            let threshold = (mean + if invert { bias } else { -bias }).clamp(24, 231) as u8;
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
    split_text_box_into_line_boxes_limited(image, bbox, 8)
}

fn split_text_box_into_line_boxes_limited(
    image: &DynamicImage,
    bbox: BoxRect,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if box_height(bbox) < 18 || box_width(bbox) < 16 {
        return Vec::new();
    }
    if max_boxes < 2 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let local_boxes = foreground_line_boxes(&crop, max_boxes);
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
    if split_boxes.len() > max_boxes {
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

fn forced_structural_split_boxes(
    image: &DynamicImage,
    bbox: BoxRect,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if !large_text_box_needs_structured_split(bbox) || max_boxes < 2 {
        return Vec::new();
    }

    let line_boxes = split_text_box_into_line_boxes_limited(image, bbox, max_boxes);
    if line_boxes.len() >= 2 {
        return line_boxes;
    }

    let mut color_boxes = split_text_box_into_color_region_boxes(image, bbox);
    if color_boxes.len() >= 2 && color_boxes.len() <= max_boxes {
        color_boxes.sort_by(reading_box_order);
        return color_boxes;
    }

    Vec::new()
}

fn large_text_box_needs_structured_split(b: BoxRect) -> bool {
    let w = box_width(b);
    let h = box_height(b).max(1);
    let aspect = w as f32 / h as f32;
    h >= 96
        || (w >= 560 && h >= 48)
        || (w >= 420 && h >= 72)
        || (aspect >= 12.0 && h >= 40)
        || box_area(b) >= 90_000
}

fn large_text_box_should_prioritize_split(b: BoxRect) -> bool {
    let w = box_width(b);
    let h = box_height(b).max(1);
    let aspect = w as f32 / h as f32;
    h >= 180 || box_area(b) >= 180_000 || (w >= 760 && h >= 72) || (aspect >= 18.0 && h >= 48)
}

fn structured_split_lines_are_plausible(b: BoxRect, split_lines: &[TextLine]) -> bool {
    if split_lines.len() < 2 {
        return false;
    }
    let split_text = split_lines
        .iter()
        .map(|line| line.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let chars = recognized_char_count(&split_text);
    if chars < split_lines.len().saturating_mul(2).max(6) {
        return false;
    }
    let avg_confidence = split_lines.iter().map(|line| line.confidence).sum::<f32>()
        / split_lines.len().max(1) as f32;
    let avg_margin = split_lines.iter().map(|line| line.avg_margin).sum::<f32>()
        / split_lines.len().max(1) as f32;
    let margin_known = split_lines
        .iter()
        .any(|line| line.avg_margin > 0.0 || line.min_margin > 0.0);
    let readable = readable_ratio(&split_text);
    let covers_vertical_space = split_lines
        .iter()
        .map(|line| box_height(line.bbox) as u64)
        .sum::<u64>()
        .saturating_mul(100)
        >= box_height(b).max(1) as u64 * 8;
    readable >= 0.58
        && avg_confidence >= 0.42
        && (!margin_known || avg_margin >= 0.015)
        && covers_vertical_space
}

fn split_text_box_vertically(
    image: &DynamicImage,
    bbox: BoxRect,
    max_boxes: usize,
) -> Vec<BoxRect> {
    if box_height(bbox) < 18 || box_width(bbox) < 16 || max_boxes == 0 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let local_boxes = foreground_line_boxes(&crop, max_boxes);
    if local_boxes.len() < 2 {
        return Vec::new();
    }

    let (img_w, img_h) = image.dimensions();
    local_boxes
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

fn split_recognized_multiline_box(
    image: &DynamicImage,
    bbox: BoxRect,
    line_count: usize,
) -> Vec<BoxRect> {
    let vertical = split_text_box_vertically(image, bbox, line_count.saturating_add(2));
    if vertical.len() == line_count {
        return vertical;
    }
    estimated_multiline_boxes(bbox, line_count)
}

fn estimated_multiline_boxes(bbox: BoxRect, line_count: usize) -> Vec<BoxRect> {
    if line_count == 0 {
        return Vec::new();
    }
    let height = box_height(bbox).max(1);
    (0..line_count)
        .map(|idx| {
            let y0 = bbox.1 + ((height as usize * idx) / line_count) as u32;
            let y1 = bbox.1 + ((height as usize * (idx + 1)) / line_count) as u32;
            (bbox.0, y0, bbox.2, y1.max(y0.saturating_add(1)).min(bbox.3))
        })
        .collect()
}

fn repair_split_boxes(image: &DynamicImage, bbox: BoxRect, budget: usize) -> Vec<BoxRect> {
    if budget < 2 {
        return Vec::new();
    }
    let mut split_boxes = split_text_box_into_line_boxes(image, bbox);
    if split_boxes.len() < 2 || split_boxes.len() > budget {
        split_boxes = split_text_box_into_color_region_boxes(image, bbox);
    }
    if split_boxes.len() >= 2 && split_boxes.len() <= budget {
        split_boxes
    } else {
        Vec::new()
    }
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
    boxes = sort_and_truncate_by(boxes, MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS, |a, b| {
        (a.1 / 8, a.0).cmp(&(b.1 / 8, b.0))
    });
    if boxes.len() < 2 {
        return Vec::new();
    }
    boxes
}

fn foreground_box_outside_boxes(image: &DynamicImage, excluded: &[BoxRect]) -> Option<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    let mask = text_foreground_mask_from_rgb(&rgb)?;

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
    let Some(mask) = text_foreground_mask_from_rgb(&rgb) else {
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

fn wide_line_segment_limit(bbox: BoxRect, budget: usize) -> usize {
    let w = box_width(bbox);
    let h = box_height(bbox).max(1);
    let aspect = w as f32 / h as f32;
    let desired = if w >= 1200 || aspect >= 28.0 {
        6
    } else if w >= 900 || aspect >= 22.0 {
        5
    } else {
        4
    };
    budget
        .min(MAX_WIDE_LINE_SEGMENTS_PER_LINE)
        .min(desired)
        .max(2)
}

fn wide_line_recognition_boxes(
    image: &DynamicImage,
    bbox: BoxRect,
    max_segments: usize,
) -> Vec<BoxRect> {
    let boxes = wide_line_segment_boxes(image, bbox, max_segments);
    if boxes.len() >= 2 {
        return boxes;
    }
    wide_line_sliding_window_boxes(image, bbox, max_segments)
}

fn wide_line_segment_boxes(
    image: &DynamicImage,
    bbox: BoxRect,
    max_segments: usize,
) -> Vec<BoxRect> {
    if max_segments < 2 {
        return Vec::new();
    }
    let bbox_w = box_width(bbox);
    let bbox_h = box_height(bbox).max(1);
    let aspect = bbox_w as f32 / bbox_h as f32;
    if bbox_w < 560 && aspect < 14.0 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let rgb = to_rgb_on_white(&crop);
    let (w_u32, h_u32) = rgb.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    let Some(mask) = text_foreground_mask_from_rgb(&rgb) else {
        return Vec::new();
    };
    let Some((min_x, min_y, max_x, max_y, foreground_count)) =
        foreground_bounds_from_mask(&mask, w, h)
    else {
        return Vec::new();
    };
    let text_w = max_x.saturating_sub(min_x).saturating_add(1);
    let text_h = max_y.saturating_sub(min_y).saturating_add(1);
    if foreground_count < 12 || text_w < 420 || text_h < 6 {
        return Vec::new();
    }

    let max_segment_w = (text_h.saturating_mul(12)).clamp(180, 360);
    let segment_count = text_w.div_ceil(max_segment_w).clamp(2, max_segments);
    if segment_count < 2 {
        return Vec::new();
    }

    let mut col_score = vec![0usize; w];
    for x in min_x..=max_x {
        let mut count = 0usize;
        for y in min_y..=max_y {
            if mask[y * w + x] {
                count += 1;
            }
        }
        col_score[x] = count;
    }

    let low_score = (text_h / 10).max(1);
    let search_radius = text_h.clamp(6, 28);
    let mut cuts = Vec::with_capacity(segment_count + 1);
    cuts.push(min_x);
    for idx in 1..segment_count {
        let target = min_x.saturating_add((text_w.saturating_mul(idx)) / segment_count);
        let left = target
            .saturating_sub(search_radius)
            .max(min_x.saturating_add(8));
        let right = target
            .saturating_add(search_radius)
            .min(max_x.saturating_sub(8));
        if right <= left {
            return Vec::new();
        }
        let mut best_x = target;
        let mut best_score = usize::MAX;
        for x in left..=right {
            let score = col_score[x];
            if score < best_score {
                best_score = score;
                best_x = x;
            }
        }
        if best_score > low_score {
            return Vec::new();
        }
        if cuts
            .last()
            .copied()
            .is_some_and(|last| best_x.saturating_sub(last) < text_h.saturating_mul(3).max(24))
        {
            return Vec::new();
        }
        cuts.push(best_x);
    }
    cuts.push(max_x.saturating_add(1));

    if cuts
        .windows(2)
        .any(|pair| pair[1].saturating_sub(pair[0]) < text_h.saturating_mul(3).max(24))
    {
        return Vec::new();
    }

    let (img_w, img_h) = image.dimensions();
    let x_pad = (text_h / 3).clamp(2, 8);
    let y_pad = (text_h / 4).clamp(1, 5);
    cuts.windows(2)
        .map(|pair| {
            clamp_box(
                (
                    bbox.0.saturating_add(pair[0].saturating_sub(x_pad) as u32),
                    bbox.1.saturating_add(min_y.saturating_sub(y_pad) as u32),
                    bbox.0
                        .saturating_add(pair[1].saturating_add(x_pad).min(w) as u32),
                    bbox.1.saturating_add(
                        max_y
                            .saturating_add(1)
                            .saturating_add(y_pad)
                            .min(h) as u32,
                    ),
                ),
                img_w,
                img_h,
            )
        })
        .collect()
}

fn wide_line_sliding_window_boxes(
    image: &DynamicImage,
    bbox: BoxRect,
    max_segments: usize,
) -> Vec<BoxRect> {
    if max_segments < 2 {
        return Vec::new();
    }
    let bbox_w = box_width(bbox);
    let bbox_h = box_height(bbox).max(1);
    let aspect = bbox_w as f32 / bbox_h as f32;
    if bbox_w < 720 && aspect < 18.0 {
        return Vec::new();
    }

    let crop = crop_box(image, bbox);
    let rgb = to_rgb_on_white(&crop);
    let (w_u32, h_u32) = rgb.dimensions();
    let w = w_u32 as usize;
    let h = h_u32 as usize;
    let Some(mask) = text_foreground_mask_from_rgb(&rgb) else {
        return Vec::new();
    };
    let Some((min_x, min_y, max_x, max_y, foreground_count)) =
        foreground_bounds_from_mask(&mask, w, h)
    else {
        return Vec::new();
    };
    let text_w = max_x.saturating_sub(min_x).saturating_add(1);
    let text_h = max_y.saturating_sub(min_y).saturating_add(1);
    if foreground_count < 16 || text_w < 560 || text_h < 6 {
        return Vec::new();
    }

    let target_window_w = (text_h.saturating_mul(15)).clamp(240, 420);
    let segment_count = text_w.div_ceil(target_window_w).clamp(2, max_segments);
    let overlap = (text_h.saturating_mul(2)).clamp(24, 64);
    let (img_w, img_h) = image.dimensions();
    let y_pad = (text_h / 4).clamp(1, 5);
    let x_pad = (text_h / 3).clamp(2, 8);

    let mut boxes = Vec::new();
    for idx in 0..segment_count {
        let start = min_x.saturating_add((text_w.saturating_mul(idx)) / segment_count);
        let end = min_x.saturating_add((text_w.saturating_mul(idx + 1)) / segment_count);
        let x0 = if idx == 0 {
            start.saturating_sub(x_pad)
        } else {
            start.saturating_sub(overlap)
        };
        let x1 = if idx + 1 == segment_count {
            end.saturating_add(x_pad)
        } else {
            end.saturating_add(overlap)
        };
        boxes.push(clamp_box(
            (
                bbox.0.saturating_add(x0.min(w) as u32),
                bbox.1.saturating_add(min_y.saturating_sub(y_pad) as u32),
                bbox.0.saturating_add(x1.min(w) as u32),
                bbox.1
                    .saturating_add(max_y.saturating_add(1).saturating_add(y_pad).min(h) as u32),
            ),
            img_w,
            img_h,
        ));
    }

    boxes.retain(|b| box_width(*b) >= text_h.saturating_mul(4).max(48) as u32);
    if boxes.len() >= 2 { boxes } else { Vec::new() }
}

fn foreground_bounds_from_mask(
    mask: &[bool],
    w: usize,
    h: usize,
) -> Option<(usize, usize, usize, usize, usize)> {
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return None;
    }
    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if count == 0 || max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some((min_x, min_y, max_x, max_y, count))
}

fn foreground_line_boxes(image: &DynamicImage, max_boxes: usize) -> Vec<BoxRect> {
    let rgb = to_rgb_on_white(image);
    let (w, h) = rgb.dimensions();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(mask) = text_foreground_mask_from_rgb(&rgb) else {
        return Vec::new();
    };
    line_boxes_from_foreground_mask(&mask, w as usize, h as usize, max_boxes)
}

fn text_foreground_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
    foreground_mask_from_rgb(rgb)
        .or_else(|| low_contrast_foreground_mask_from_rgb(rgb))
        .or_else(|| soft_color_foreground_mask_from_rgb(rgb))
        .or_else(|| dark_luma_mask_from_rgb(rgb))
}

fn low_contrast_foreground_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
    let binary = low_contrast_binary_luma_from_rgb(rgb)?;
    Some(binary.pixels().map(|pixel| pixel[0] < 128).collect())
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

fn soft_color_foreground_mask_from_rgb(rgb: &image::RgbImage) -> Option<Vec<bool>> {
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
    if max_distance < 8 {
        return None;
    }

    let threshold = otsu_threshold_values(&distances).clamp(8, 64);
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
    if foreground_count < 4 || !(0.001..=0.58).contains(&foreground_ratio) {
        return None;
    }
    if !foreground_glyph_textness_score(&mask, w as usize, h as usize)
        .is_some_and(|score| score >= -20)
    {
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

    boxes = sort_and_truncate_by(boxes, max_boxes, |a, b| (a.1, a.0).cmp(&(b.1, b.0)));
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

fn ctc_decode_with_stats(
    logits: &[f32],
    out_shape: &[usize],
    alphabet: &[String],
) -> (String, f32, CtcDecodeStats) {
    let greedy = ctc_greedy_decode_with_stats(logits, out_shape, alphabet);
    let Some(beam) = ctc_path_beam_decode_with_stats(logits, out_shape, alphabet) else {
        return greedy;
    };
    if beam.0 == greedy.0 {
        return greedy;
    }
    let greedy_quality = recognition_text_quality_with_margin(
        &greedy.0,
        greedy.1,
        greedy.2.avg_margin,
        greedy.2.min_margin,
    );
    let beam_quality =
        recognition_text_quality_with_margin(&beam.0, beam.1, beam.2.avg_margin, beam.2.min_margin);
    if greedy.0.trim().is_empty()
        || (beam_quality > greedy_quality + 5.0 && beam.1 + 0.04 >= greedy.1)
    {
        beam
    } else {
        greedy
    }
}

fn ctc_greedy_decode_with_stats(
    logits: &[f32],
    out_shape: &[usize],
    alphabet: &[String],
) -> (String, f32, CtcDecodeStats) {
    let shape = g_outer_shape(logits, out_shape);
    if shape.len() < 3 {
        return (String::new(), 0.0, CtcDecodeStats::default());
    }

    let (steps, classes, channel_first) = if shape[1] > shape[2] {
        (shape[2], shape[1], true)
    } else {
        (shape[1], shape[2], false)
    };

    if classes <= 1 {
        return (String::new(), 0.0, CtcDecodeStats::default());
    }

    let blank_id = 0usize;
    let mut prev = blank_id;
    let mut text = String::new();
    let mut prob_sum = 0.0f32;
    let mut margin_sum = 0.0f32;
    let mut min_margin = f32::INFINITY;
    let mut min_char_confidence = f32::INFINITY;
    let mut count = 0usize;

    for t in 0..steps {
        let (best_id, best_prob, second_prob) =
            ctc_frame_top2_probability(logits, t, steps, classes, channel_first);
        if best_id != blank_id && best_id != prev {
            let idx = best_id.saturating_sub(1);
            if let Some(ch) = alphabet.get(idx) {
                if ch == "\u{3000}" {
                    continue;
                }
                text.push_str(ch);
                prob_sum += best_prob;
                min_char_confidence = min_char_confidence.min(best_prob);
                let margin = (best_prob - second_prob).max(0.0);
                margin_sum += margin;
                min_margin = min_margin.min(margin);
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
    let stats = if count == 0 {
        CtcDecodeStats::default()
    } else {
        CtcDecodeStats {
            avg_margin: margin_sum / count as f32,
            min_margin,
            char_min_confidence: if min_char_confidence.is_finite() {
                min_char_confidence
            } else {
                0.0
            },
        }
    };
    (text, confidence, stats)
}

const CTC_LOG_ZERO: f32 = -1.0e9;

#[derive(Clone)]
struct CtcPrefixState {
    p_blank: f32,
    p_non_blank: f32,
    best_blank_path: Vec<usize>,
    best_blank_score: f32,
    best_non_blank_path: Vec<usize>,
    best_non_blank_score: f32,
}

fn ctc_path_beam_decode_with_stats(
    logits: &[f32],
    out_shape: &[usize],
    alphabet: &[String],
) -> Option<(String, f32, CtcDecodeStats)> {
    let shape = g_outer_shape(logits, out_shape);
    if shape.len() < 3 {
        return None;
    }
    let (steps, classes, channel_first) = if shape[1] > shape[2] {
        (shape[2], shape[1], true)
    } else {
        (shape[1], shape[2], false)
    };
    if steps == 0 || classes <= 1 {
        return None;
    }

    let mut beams: HashMap<Vec<usize>, CtcPrefixState> = HashMap::new();
    beams.insert(
        Vec::new(),
        CtcPrefixState {
            p_blank: 0.0,
            p_non_blank: CTC_LOG_ZERO,
            best_blank_path: Vec::new(),
            best_blank_score: 0.0,
            best_non_blank_path: Vec::new(),
            best_non_blank_score: CTC_LOG_ZERO,
        },
    );
    for t in 0..steps {
        let top = ctc_top_classes(logits, t, steps, classes, channel_first, CTC_TOP_K);
        let mut active = beams.into_iter().collect::<Vec<_>>();
        active.sort_by(|a, b| compare_ctc_prefix_states(&a.1, &b.1));
        active.truncate(CTC_BEAM_SIZE);

        let mut next: HashMap<Vec<usize>, CtcPrefixState> = HashMap::new();
        for (prefix, state) in active {
            let total_score = ctc_prefix_total_score(&state);
            let (best_path, best_path_score) = state.best_path_and_score();
            for &(class_id, score) in &top {
                if class_id == 0 {
                    let mut path = best_path.clone();
                    path.push(0);
                    ctc_update_blank_state(
                        next.entry(prefix.clone())
                            .or_insert_with(CtcPrefixState::empty),
                        total_score + score,
                        path,
                        best_path_score + score,
                    );
                    continue;
                }

                let last = prefix.last().copied();
                if last == Some(class_id) {
                    if state.p_non_blank > CTC_LOG_ZERO / 2.0 {
                        let mut path = state.best_non_blank_path.clone();
                        path.push(class_id);
                        ctc_update_non_blank_state(
                            next.entry(prefix.clone())
                                .or_insert_with(CtcPrefixState::empty),
                            state.p_non_blank + score,
                            path,
                            state.best_non_blank_score + score,
                        );
                    }
                    if state.p_blank > CTC_LOG_ZERO / 2.0 {
                        let mut extended = prefix.clone();
                        extended.push(class_id);
                        let mut path = state.best_blank_path.clone();
                        path.push(class_id);
                        ctc_update_non_blank_state(
                            next.entry(extended).or_insert_with(CtcPrefixState::empty),
                            state.p_blank + score,
                            path,
                            state.best_blank_score + score,
                        );
                    }
                } else {
                    let mut extended = prefix.clone();
                    extended.push(class_id);
                    let mut path = best_path.clone();
                    path.push(class_id);
                    ctc_update_non_blank_state(
                        next.entry(extended).or_insert_with(CtcPrefixState::empty),
                        total_score + score,
                        path,
                        best_path_score + score,
                    );
                }
            }
        }
        let mut kept = next.into_iter().collect::<Vec<_>>();
        kept.sort_by(|a, b| compare_ctc_prefix_states(&a.1, &b.1));
        kept.truncate(CTC_BEAM_SIZE);
        beams = kept.into_iter().collect();
    }

    let mut collapsed: Vec<(String, f32, Vec<usize>)> = Vec::new();
    for (prefix, state) in beams {
        let text = ctc_prefix_to_text(&prefix, alphabet);
        if text.trim().is_empty() {
            continue;
        }
        let score = ctc_prefix_total_score(&state);
        let (path, _) = state.best_path_and_score();
        if collapsed
            .iter()
            .any(|(existing, existing_score, _)| existing == &text && *existing_score >= score)
        {
            continue;
        }
        collapsed
            .retain(|(existing, existing_score, _)| existing != &text || *existing_score > score);
        collapsed.push((text, score, path));
    }
    collapsed.sort_by(|a, b| total_cmp_desc(a.1, b.1));
    let (_, _, ids) = collapsed.first()?;
    Some(ctc_stats_for_path(
        logits,
        ids,
        steps,
        classes,
        channel_first,
        alphabet,
    ))
}

impl CtcPrefixState {
    fn empty() -> Self {
        Self {
            p_blank: CTC_LOG_ZERO,
            p_non_blank: CTC_LOG_ZERO,
            best_blank_path: Vec::new(),
            best_blank_score: CTC_LOG_ZERO,
            best_non_blank_path: Vec::new(),
            best_non_blank_score: CTC_LOG_ZERO,
        }
    }

    fn best_path_and_score(&self) -> (Vec<usize>, f32) {
        if self.best_blank_score >= self.best_non_blank_score {
            (self.best_blank_path.clone(), self.best_blank_score)
        } else {
            (self.best_non_blank_path.clone(), self.best_non_blank_score)
        }
    }
}

fn ctc_prefix_total_score(state: &CtcPrefixState) -> f32 {
    log_sum_exp2(state.p_blank, state.p_non_blank)
}

#[inline]
fn compare_ctc_prefix_states(a: &CtcPrefixState, b: &CtcPrefixState) -> std::cmp::Ordering {
    total_cmp_desc(ctc_prefix_total_score(a), ctc_prefix_total_score(b))
}

#[inline]
fn total_cmp_desc(lhs: f32, rhs: f32) -> std::cmp::Ordering {
    let lhs = if lhs == -0.0 { 0.0 } else { lhs };
    let rhs = if rhs == -0.0 { 0.0 } else { rhs };
    rhs.total_cmp(&lhs)
}

fn ctc_update_blank_state(
    state: &mut CtcPrefixState,
    score: f32,
    path: Vec<usize>,
    path_score: f32,
) {
    state.p_blank = log_sum_exp2(state.p_blank, score);
    if path_score > state.best_blank_score {
        state.best_blank_score = path_score;
        state.best_blank_path = path;
    }
}

fn ctc_update_non_blank_state(
    state: &mut CtcPrefixState,
    score: f32,
    path: Vec<usize>,
    path_score: f32,
) {
    state.p_non_blank = log_sum_exp2(state.p_non_blank, score);
    if path_score > state.best_non_blank_score {
        state.best_non_blank_score = path_score;
        state.best_non_blank_path = path;
    }
}

fn log_sum_exp2(a: f32, b: f32) -> f32 {
    if a <= CTC_LOG_ZERO / 2.0 {
        return b;
    }
    if b <= CTC_LOG_ZERO / 2.0 {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn ctc_prefix_to_text(prefix: &[usize], alphabet: &[String]) -> String {
    let mut text = String::new();
    for id in prefix {
        let idx = id.saturating_sub(1);
        if let Some(ch) = alphabet.get(idx)
            && ch != "\u{3000}"
        {
            text.push_str(ch);
        }
    }
    text
}

fn ctc_top_classes(
    logits: &[f32],
    t: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
    k: usize,
) -> Vec<(usize, f32)> {
    let log_probs = ctc_frame_log_probabilities(logits, t, steps, classes, channel_first);
    let mut top = Vec::<(usize, f32)>::new();
    for class_id in 0..classes {
        let value = log_probs
            .get(class_id)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        let pos = top
            .iter()
            .position(|(_, seen)| value > *seen)
            .unwrap_or(top.len());
        if pos < k {
            top.insert(pos, (class_id, value));
            top.truncate(k);
        }
    }
    if !top.iter().any(|(class_id, _)| *class_id == 0) {
        top.push((0, log_probs.first().copied().unwrap_or(f32::NEG_INFINITY)));
    }
    top
}

fn ctc_frame_top2_probability(
    logits: &[f32],
    t: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
) -> (usize, f32, f32) {
    let log_probs = ctc_frame_log_probabilities(logits, t, steps, classes, channel_first);
    let mut best_id = 0usize;
    let mut best_log = f32::NEG_INFINITY;
    let mut second_log = f32::NEG_INFINITY;
    for (class_id, value) in log_probs.iter().copied().enumerate() {
        if value > best_log {
            second_log = best_log;
            best_log = value;
            best_id = class_id;
        } else if value > second_log {
            second_log = value;
        }
    }
    (
        best_id,
        if best_log.is_finite() {
            best_log.exp()
        } else {
            0.0
        },
        if second_log.is_finite() {
            second_log.exp()
        } else {
            0.0
        },
    )
}

fn ctc_frame_probability(
    logits: &[f32],
    t: usize,
    class_id: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
) -> f32 {
    ctc_frame_log_probabilities(logits, t, steps, classes, channel_first)
        .get(class_id)
        .copied()
        .filter(|value| value.is_finite())
        .map(f32::exp)
        .unwrap_or(0.0)
}

fn ctc_frame_log_probabilities(
    logits: &[f32],
    t: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
) -> Vec<f32> {
    if classes == 0 {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(classes);
    let mut sum = 0.0f32;
    let mut all_probability_like = true;
    let mut max_value = f32::NEG_INFINITY;
    for class_id in 0..classes {
        let value = ctc_frame_value(logits, t, class_id, steps, classes, channel_first);
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            all_probability_like = false;
        }
        if value.is_finite() {
            sum += value.max(0.0);
            max_value = max_value.max(value);
        }
        values.push(value);
    }

    if all_probability_like && sum > 1.0e-6 {
        return values
            .into_iter()
            .map(|value| (value / sum).max(1.0e-6).ln())
            .collect();
    }
    if !max_value.is_finite() {
        return vec![CTC_LOG_ZERO; classes];
    }

    let mut exp_sum = 0.0f32;
    let mut exp_values = Vec::with_capacity(classes);
    for value in values {
        let exp = if value.is_finite() {
            (value - max_value).exp()
        } else {
            0.0
        };
        exp_sum += exp;
        exp_values.push(exp);
    }
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        return vec![CTC_LOG_ZERO; classes];
    }
    exp_values
        .into_iter()
        .map(|value| (value / exp_sum).max(1.0e-6).ln())
        .collect()
}

fn ctc_frame_value(
    logits: &[f32],
    t: usize,
    class_id: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
) -> f32 {
    let idx = if channel_first {
        class_id.saturating_mul(steps).saturating_add(t)
    } else {
        t.saturating_mul(classes).saturating_add(class_id)
    };
    logits.get(idx).copied().unwrap_or(f32::NEG_INFINITY)
}

fn ctc_stats_for_path(
    logits: &[f32],
    ids: &[usize],
    steps: usize,
    classes: usize,
    channel_first: bool,
    alphabet: &[String],
) -> (String, f32, CtcDecodeStats) {
    let mut prev = 0usize;
    let mut text = String::new();
    let mut prob_sum = 0.0f32;
    let mut margin_sum = 0.0f32;
    let mut min_margin = f32::INFINITY;
    let mut min_char_confidence = f32::INFINITY;
    let mut count = 0usize;
    for (t, id) in ids.iter().copied().enumerate().take(steps) {
        if id != 0 && id != prev {
            let idx = id.saturating_sub(1);
            if let Some(ch) = alphabet.get(idx) {
                if ch == "\u{3000}" {
                    prev = id;
                    continue;
                }
                text.push_str(ch);
                let value = ctc_frame_probability(logits, t, id, steps, classes, channel_first);
                let second =
                    ctc_second_best_probability(logits, t, id, steps, classes, channel_first);
                prob_sum += value;
                min_char_confidence = min_char_confidence.min(value);
                let margin = (value - second).max(0.0);
                margin_sum += margin;
                min_margin = min_margin.min(margin);
                count += 1;
            }
        }
        prev = id;
    }
    if count == 0 {
        return (String::new(), 0.0, CtcDecodeStats::default());
    }
    (
        text,
        prob_sum / count as f32,
        CtcDecodeStats {
            avg_margin: margin_sum / count as f32,
            min_margin,
            char_min_confidence: if min_char_confidence.is_finite() {
                min_char_confidence
            } else {
                0.0
            },
        },
    )
}

fn ctc_second_best_probability(
    logits: &[f32],
    t: usize,
    selected: usize,
    steps: usize,
    classes: usize,
    channel_first: bool,
) -> f32 {
    let log_probs = ctc_frame_log_probabilities(logits, t, steps, classes, channel_first);
    let mut second = f32::NEG_INFINITY;
    for class_id in 0..classes {
        if class_id == selected {
            continue;
        }
        second = second.max(
            log_probs
                .get(class_id)
                .copied()
                .unwrap_or(f32::NEG_INFINITY),
        );
    }
    if second.is_finite() {
        second.exp()
    } else {
        0.0
    }
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
    if recognition_candidate_model_score(&alt) > recognition_candidate_model_score(&primary) + 6.0 {
        return alt;
    }
    primary
}

fn recognition_candidate_model_score(candidate: &RecCandidate) -> f32 {
    recognition_text_quality_with_margin(
        &candidate.text,
        candidate.confidence,
        candidate.avg_margin,
        candidate.min_margin,
    )
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
    if candidate.confidence < 0.70 && is_low_value_ascii_noise(text) {
        return false;
    }
    true
}

#[inline]
fn is_stable_short_text_candidate(candidate: &RecCandidate, text: &str, max_chars: usize) -> bool {
    let text_len = recognized_char_count(text);
    if text_len == 0 || text_len > max_chars {
        return false;
    }
    if candidate.confidence < STABLE_TEXT_CONFIDENCE {
        return false;
    }
    if candidate.avg_margin < STABLE_TEXT_AVG_MARGIN {
        return false;
    }
    if candidate.min_margin < STABLE_TEXT_MIN_MARGIN {
        return false;
    }
    if readable_ratio(text) < STABLE_TEXT_READABLE_RATIO {
        return false;
    }
    if dominant_char_ratio(text) > STABLE_TEXT_DOMINANT_RATIO {
        return false;
    }
    if punctuation_ratio(text) > STABLE_TEXT_PUNCTUATION_RATIO {
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
    fn enhancement_variants_limited_stops_at_budget_even_with_full_capacity() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 24, Luma([220])));
        let variants = enhancement_variants_limited(&img, 1);

        assert_eq!(variants.len(), 1);
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
    fn contour_refinement_splits_sparse_multiline_component() {
        let w = 80usize;
        let h = 48usize;
        let mut mask = vec![false; w * h];
        for y in 8..12 {
            for x in 10..68 {
                mask[y * w + x] = true;
            }
        }
        for y in 30..34 {
            for x in 14..72 {
                mask[y * w + x] = true;
            }
        }

        let boxes = contour_refined_boxes_from_component(&mask, 0, 0, 79, 47, w, h, 4);

        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].1 <= 8 && boxes[0].3 >= 12);
        assert!(boxes[1].1 <= 30 && boxes[1].3 >= 34);
    }

    #[test]
    fn nms_boxes_removes_overlapping_boxes() {
        let boxes = vec![(0, 0, 12, 12), (1, 1, 11, 11), (30, 0, 40, 10)];
        let kept = nms_boxes(boxes, 0.35);
        assert_eq!(kept, vec![(0, 0, 12, 12), (30, 0, 40, 10)]);
    }

    #[test]
    fn dedupe_box_candidates_removes_nested_overlaps() {
        let boxes = vec![
            (0, 0, 120, 28),
            (0, 0, 118, 26),
            (60, 0, 140, 24),
            (30, 4, 90, 20),
        ];
        let kept = dedupe_box_candidates(boxes);
        assert_eq!(kept, vec![(0, 0, 120, 28), (60, 0, 140, 24)]);
    }

    #[test]
    fn raw_split_detection_candidates_keep_merged_and_parts() {
        let rgb = image::RgbImage::from_pixel(128, 24, image::Rgb([255, 255, 255]));
        let raw = vec![(0, 0, 40, 12), (46, 0, 104, 12)];
        let boxes = merge_nearby_detection_boxes(&rgb, raw.clone());
        assert_eq!(boxes, vec![(0, 0, 104, 12)]);

        let alternatives = raw_split_detection_candidates(boxes[0], &raw);

        assert_eq!(alternatives.len(), 2);
        assert!(alternatives.contains(&(0, 0, 40, 12)));
        assert!(alternatives.contains(&(46, 0, 104, 12)));
    }

    #[test]
    fn merge_nearby_detection_boxes_respects_visual_gutter() {
        let rgb = image::RgbImage::from_pixel(140, 32, image::Rgb([245, 245, 245]));
        let raw = vec![(0, 8, 42, 22), (74, 8, 120, 22)];

        let boxes = merge_nearby_detection_boxes(&rgb, raw);

        assert_eq!(boxes, vec![(0, 8, 42, 22), (74, 8, 120, 22)]);
    }

    #[test]
    fn merge_nearby_detection_boxes_respects_small_panel_gap() {
        let rgb = image::RgbImage::from_pixel(180, 32, image::Rgb([250, 250, 250]));
        let raw = vec![(10, 8, 70, 22), (86, 8, 148, 22)];

        let boxes = merge_nearby_detection_boxes(&rgb, raw);

        assert_eq!(boxes, vec![(10, 8, 70, 22), (86, 8, 148, 22)]);
    }

    #[test]
    fn page_region_boxes_split_detected_columns() {
        let boxes = vec![
            detection_box((28, 20, 150, 38)),
            detection_box((34, 90, 170, 108)),
            detection_box((42, 180, 160, 198)),
            detection_box((270, 32, 410, 50)),
            detection_box((278, 112, 430, 130)),
            detection_box((286, 220, 420, 238)),
            detection_box((710, 42, 900, 60)),
            detection_box((724, 140, 880, 158)),
        ];

        let regions = page_region_boxes_from_detection_boxes(&boxes, (1000, 600));

        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].1, 0);
        assert_eq!(regions[0].3, 600);
        assert!(regions[0].2 <= regions[1].0);
        assert!(regions[1].2 <= regions[2].0);
    }

    #[test]
    fn visual_page_region_boxes_split_edge_rich_columns() {
        let mut rgb = image::RgbImage::from_pixel(900, 240, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture(&mut rgb, 30, 220);
        draw_synthetic_text_texture(&mut rgb, 330, 530);
        draw_synthetic_text_texture(&mut rgb, 650, 840);

        let regions = visual_page_region_boxes(&DynamicImage::ImageRgb8(rgb));

        assert_eq!(regions.len(), 3);
        assert!(regions[0].2 <= regions[1].0);
        assert!(regions[1].2 <= regions[2].0);
    }

    #[test]
    fn visual_page_region_boxes_skip_single_dense_column() {
        let mut rgb = image::RgbImage::from_pixel(900, 240, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture(&mut rgb, 40, 860);

        let regions = visual_page_region_boxes(&DynamicImage::ImageRgb8(rgb));

        assert!(regions.is_empty());
    }

    #[test]
    fn page_region_boxes_can_use_visual_candidates_without_detection_boxes() {
        let mut rgb = image::RgbImage::from_pixel(900, 240, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture(&mut rgb, 30, 220);
        draw_synthetic_text_texture(&mut rgb, 330, 530);
        draw_synthetic_text_texture(&mut rgb, 650, 840);

        let regions = page_region_boxes(&DynamicImage::ImageRgb8(rgb), &[]);

        assert_eq!(regions.len(), 3);
    }

    #[test]
    fn visual_page_region_boxes_can_split_two_dimensional_panels() {
        let mut rgb = image::RgbImage::from_pixel(1000, 700, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture_at(&mut rgb, 60, 360, 54, 7);
        draw_synthetic_text_texture_at(&mut rgb, 610, 930, 54, 7);
        draw_synthetic_text_texture_at(&mut rgb, 60, 360, 430, 7);
        draw_synthetic_text_texture_at(&mut rgb, 610, 930, 430, 7);

        let regions = visual_page_region_boxes(&DynamicImage::ImageRgb8(rgb));

        assert_eq!(regions.len(), 4);
        assert!(regions.iter().any(|b| b.0 < 200 && b.1 < 200));
        assert!(regions.iter().any(|b| b.0 > 500 && b.1 < 200));
        assert!(regions.iter().any(|b| b.0 < 200 && b.1 > 350));
        assert!(regions.iter().any(|b| b.0 > 500 && b.1 > 350));
    }

    #[test]
    fn uncovered_visual_text_boxes_skip_reliably_covered_lines() {
        let mut rgb = image::RgbImage::from_pixel(360, 120, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture_at(&mut rgb, 24, 140, 24, 4);
        draw_synthetic_text_texture_at(&mut rgb, 220, 340, 24, 4);
        let existing = vec![OcrTextRegion {
            bbox: [18, 18, 150, 100],
            text: "Alpha Beta".to_string(),
            confidence: 0.88,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [18, 18, 150, 100],
                text: "Alpha Beta".to_string(),
                confidence: 0.88,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.88,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];

        let boxes = uncovered_visual_text_boxes(&DynamicImage::ImageRgb8(rgb), &existing);

        assert!(!boxes.is_empty());
        assert!(boxes.iter().all(|b| b.0 > 180));
    }

    #[test]
    fn uncovered_visual_supplement_runs_when_no_regions_exist() {
        let cfg = OcrConfig::default();
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(320, 80, Luma([255])));

        assert!(should_use_uncovered_visual_supplement(
            &img,
            &cfg,
            "",
            0,
            0,
            &[]
        ));
        assert!(should_use_uncovered_visual_supplement(
            &img,
            &cfg,
            "",
            2,
            0,
            &[]
        ));
    }

    #[test]
    fn supplement_pass_progress_detects_meaningful_gain() {
        assert!(supplement_pass_made_progress(
            "Alpha",
            0.61,
            1,
            "AlphaBeta",
            0.61,
            1
        ));
        assert!(supplement_pass_made_progress(
            "Alpha", 0.61, 1, "Alfa", 0.63, 1
        ));
        assert!(supplement_pass_made_progress(
            "Alpha", 0.61, 1, "Alpha", 0.63, 2
        ));
        assert!(!supplement_pass_made_progress(
            "Alpha", 0.61, 1, "Alpha", 0.61, 1
        ));
    }

    #[test]
    fn supplement_pass_continue_requires_progress_or_repairable_lines() {
        let weak_regions = vec![ocr_region_with_line(
            [0, 0, 120, 24],
            "WeakText",
            0.38,
            "det",
        )];
        let strong_regions = vec![ocr_region_with_line(
            [0, 0, 120, 24],
            "StrongText",
            0.92,
            "det",
        )];

        assert!(should_continue_eager_supplement_pass(
            "Alpha",
            0.61,
            4,
            1,
            &weak_regions,
            "Alpha",
            0.61,
            1,
        ));
        assert!(!should_continue_eager_supplement_pass(
            "Alpha",
            0.61,
            4,
            1,
            &strong_regions,
            "Alpha",
            0.61,
            1,
        ));
    }

    #[test]
    fn should_continue_eager_supplements_short_circuit_for_high_confidence() {
        assert!(!should_continue_eager_supplements(
            "AlphaBeta",
            0.96,
            4,
            2,
            &[]
        ));
    }

    #[test]
    fn color_region_det_candidates_keep_large_partially_covered_panel() {
        let mut rgb = image::RgbImage::from_pixel(320, 160, image::Rgb([255, 255, 255]));
        fill_rect(&mut rgb, (20, 20, 220, 120), image::Rgb([232, 239, 248]));
        draw_synthetic_text_texture_at(&mut rgb, 34, 150, 34, 4);
        draw_synthetic_text_texture_at(&mut rgb, 34, 180, 82, 4);
        let existing = vec![ocr_region_with_line(
            [30, 32, 152, 54],
            "Alpha Beta",
            0.88,
            "det",
        )];

        let (_, boxes) =
            color_region_det_candidate_boxes(&DynamicImage::ImageRgb8(rgb), &existing, 16);

        assert!(
            boxes
                .iter()
                .any(|b| b.0 <= 24 && b.1 <= 24 && b.2 >= 216 && b.3 >= 116)
        );
    }

    #[test]
    fn color_region_det_candidates_skip_covered_line_like_panel() {
        let mut rgb = image::RgbImage::from_pixel(240, 100, image::Rgb([255, 255, 255]));
        fill_rect(&mut rgb, (20, 30, 200, 58), image::Rgb([232, 239, 248]));
        draw_synthetic_text_texture_at(&mut rgb, 34, 150, 36, 4);
        let existing = vec![ocr_region_with_line(
            [30, 32, 152, 56],
            "Alpha Beta",
            0.88,
            "det",
        )];

        let (_, boxes) =
            color_region_det_candidate_boxes(&DynamicImage::ImageRgb8(rgb), &existing, 16);

        assert!(
            boxes
                .iter()
                .all(|b| !(b.0 <= 24 && b.1 <= 34 && b.2 >= 196 && b.3 >= 54))
        );
    }

    #[test]
    fn page_region_sources_get_local_repair_budgets() {
        assert_eq!(
            split_line_recognition_budget("det"),
            MAX_SPLIT_LINE_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            line_repair_recognition_budget("det"),
            MAX_LINE_REPAIR_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            split_line_recognition_budget("page-region:1"),
            MAX_PAGE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            line_repair_recognition_budget("page-region:1"),
            MAX_PAGE_REGION_REPAIR_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            split_line_recognition_budget("color-region-det:eager:1"),
            MAX_COLOR_REGION_DET_SPLIT_LINE_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            line_repair_recognition_budget("color-region-det:eager:1"),
            MAX_COLOR_REGION_DET_REPAIR_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            split_line_recognition_budget("tile-region:1"),
            MAX_TILE_REGION_SPLIT_LINE_RECOGNITIONS_PER_PASS
        );
        assert_eq!(
            line_repair_recognition_budget("tile-region:1"),
            MAX_TILE_REGION_REPAIR_RECOGNITIONS_PER_PASS
        );
    }

    #[test]
    fn high_res_tile_supplement_requires_large_or_weak_input() {
        let cfg = OcrConfig::default();
        let small = DynamicImage::ImageLuma8(GrayImage::from_pixel(480, 320, Luma([255])));
        let large = DynamicImage::ImageLuma8(GrayImage::from_pixel(1200, 900, Luma([255])));
        let strong_regions = vec![OcrTextRegion {
            bbox: [10, 10, 180, 36],
            text: "Alpha Beta".to_string(),
            confidence: 0.88,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 180, 36],
                text: "Alpha Beta".to_string(),
                confidence: 0.88,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.88,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];
        let weak_regions = vec![OcrTextRegion {
            bbox: [10, 10, 640, 92],
            text: "Alpha Beta".to_string(),
            confidence: 0.70,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 640, 92],
                text: "Alpha Beta".to_string(),
                confidence: 0.70,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.70,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];

        assert!(!should_use_high_res_tile_supplement(
            &small,
            &cfg,
            "Alpha Beta",
            0.30,
            12,
            2,
            &weak_regions
        ));
        assert!(!should_use_high_res_tile_supplement(
            &large,
            &cfg,
            "Alpha Beta",
            0.88,
            6,
            5,
            &strong_regions
        ));
        assert!(should_use_high_res_tile_supplement(
            &large,
            &cfg,
            "Alpha Beta",
            0.88,
            6,
            5,
            &weak_regions
        ));
        assert!(should_use_high_res_tile_supplement(
            &large,
            &cfg,
            "",
            0.0,
            0,
            0,
            &[]
        ));
    }

    #[test]
    fn reading_region_order_is_total_for_staggered_layouts() {
        let a = LayoutRegion {
            bbox: (340, 48, 520, 96),
            lines: Vec::new(),
        };
        let b = LayoutRegion {
            bbox: (24, 72, 260, 124),
            lines: Vec::new(),
        };
        let c = LayoutRegion {
            bbox: (320, 128, 560, 188),
            lines: Vec::new(),
        };

        let mut regions = vec![c.clone(), a.clone(), b.clone()];
        regions.sort_by(reading_region_order);

        let keys = regions.iter().map(reading_region_key).collect::<Vec<_>>();
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(reading_region_order(&a, &a), std::cmp::Ordering::Equal);
        assert_eq!(reading_region_order(&b, &b), std::cmp::Ordering::Equal);
        assert_eq!(reading_region_order(&c, &c), std::cmp::Ordering::Equal);
    }

    #[test]
    fn tile_axis_starts_cover_tail_with_overlap() {
        let starts = tile_axis_starts(1500, 640, 128);

        assert_eq!(starts.first().copied(), Some(0));
        assert_eq!(starts.last().copied(), Some(860));
        assert!(starts.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn high_res_tile_boxes_keep_limited_textured_tiles() {
        let mut rgb = image::RgbImage::from_pixel(1500, 900, image::Rgb([255, 255, 255]));
        draw_synthetic_text_texture(&mut rgb, 40, 520);
        draw_synthetic_text_texture(&mut rgb, 820, 1420);

        let tiles = high_res_tile_boxes(&DynamicImage::ImageRgb8(rgb), 960);

        assert!(!tiles.is_empty());
        assert!(tiles.len() <= MAX_HIGH_RES_TILE_DET_PASSES);
        assert!(
            tiles
                .iter()
                .all(|b| box_width(*b) <= 640 && box_height(*b) <= 640)
        );
        assert!(
            tiles
                .windows(2)
                .all(|pair| reading_box_order(&pair[0], &pair[1]) != std::cmp::Ordering::Greater)
        );
    }

    #[test]
    fn page_region_boxes_skip_small_or_simple_images() {
        let boxes = vec![
            detection_box((28, 20, 150, 38)),
            detection_box((34, 90, 170, 108)),
        ];

        let regions = page_region_boxes_from_detection_boxes(&boxes, (1000, 600));

        assert!(regions.is_empty());
    }

    #[test]
    fn bbox_offset_transform_maps_crop_coordinates() {
        let mapped = BboxTransform::Offset {
            dx: 100,
            dy: 40,
            max_w: 300,
            max_h: 120,
        }
        .map_box((10, 6, 80, 30));

        assert_eq!(mapped, (110, 46, 180, 70));
    }

    #[test]
    fn merge_recognized_line_sets_dedupes_existing_lines() {
        let current = vec![OcrTextRegion {
            bbox: [10, 10, 120, 28],
            text: "Header".to_string(),
            confidence: 0.64,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 120, 28],
                text: "Header".to_string(),
                confidence: 0.64,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.64,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];
        let mut candidate_lines = vec![
            text_line((12, 11, 122, 29), "Header", 0.78),
            text_line((220, 48, 360, 66), "New detail", 0.72),
        ];
        let candidate = recognized_from_text_lines(&mut candidate_lines);

        let merged = merge_recognized_line_sets(&current, &candidate);

        assert_eq!(merged.line_count, 2);
        assert_eq!(merged.text.matches("Header").count(), 1);
        assert!(merged.text.contains("New detail"));
    }

    #[test]
    fn page_region_supplement_keeps_only_strong_new_lines() {
        let current = vec![
            OcrTextRegion {
                bbox: [10, 10, 120, 28],
                text: "Header".to_string(),
                confidence: 0.64,
                source: "det".to_string(),
                lines: vec![OcrTextLine {
                    bbox: [10, 10, 120, 28],
                    text: "Header".to_string(),
                    confidence: 0.64,
                    avg_margin: 0.10,
                    min_margin: 0.06,
                    char_min_confidence: 0.64,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                }],
            },
            OcrTextRegion {
                bbox: [10, 40, 220, 60],
                text: "WidgetSuite2408".to_string(),
                confidence: 0.70,
                source: "det".to_string(),
                lines: vec![OcrTextLine {
                    bbox: [10, 40, 220, 60],
                    text: "WidgetSuite2408".to_string(),
                    confidence: 0.70,
                    avg_margin: 0.10,
                    min_margin: 0.06,
                    char_min_confidence: 0.70,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                }],
            },
        ];
        let mut candidate_lines = vec![
            text_line((12, 11, 122, 29), "Header", 0.78),
            text_line((220, 48, 360, 66), "Approve", 0.72),
            text_line((420, 48, 450, 66), "X", 0.92),
            text_line((500, 48, 560, 66), "2408", 0.88),
            text_line((600, 48, 720, 66), "WidgetSuite@", 0.88),
        ];
        let candidate = recognized_from_text_lines(&mut candidate_lines);

        let supplement = filter_page_region_supplement(&candidate, &current);

        assert_eq!(supplement.text, "Approve");
        assert_eq!(supplement.line_count, 1);
    }

    #[test]
    fn ocr_trace_lines_include_region_line_source_and_crop_size() {
        let regions = vec![OcrTextRegion {
            bbox: [10, 20, 120, 60],
            text: "Alpha\nBeta".to_string(),
            confidence: 0.73,
            source: "det".to_string(),
            lines: vec![
                OcrTextLine {
                    bbox: [10, 20, 80, 36],
                    text: "Alpha".to_string(),
                    confidence: 0.70,
                    avg_margin: 0.10,
                    min_margin: 0.06,
                    char_min_confidence: 0.70,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                },
                OcrTextLine {
                    bbox: [12, 42, 120, 60],
                    text: "Beta".to_string(),
                    confidence: 0.76,
                    avg_margin: 0.12,
                    min_margin: 0.07,
                    char_min_confidence: 0.76,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "page-region:1".to_string(),
                },
            ],
        }];

        let lines = ocr_trace_lines_from_regions(&regions);

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].region_index, 0);
        assert_eq!(lines[0].line_index, 0);
        assert_eq!(lines[0].bbox, [10, 20, 80, 36]);
        assert_eq!(lines[0].crop_size, [70, 16]);
        assert_eq!(lines[0].avg_margin, 0.10);
        assert_eq!(lines[0].min_margin, 0.06);
        assert_eq!(lines[0].char_min_confidence, 0.70);
        assert_eq!(lines[0].support_count, 1);
        assert_eq!(lines[1].source, "page-region:1");
    }

    #[test]
    fn ocr_trace_json_escapes_text_and_records_lines() {
        let regions = vec![OcrTextRegion {
            bbox: [0, 0, 80, 24],
            text: "A \"quoted\" line".to_string(),
            confidence: 0.82,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [0, 0, 80, 24],
                text: "A \"quoted\" line".to_string(),
                confidence: 0.82,
                avg_margin: 0.11,
                min_margin: 0.05,
                char_min_confidence: 0.82,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];
        let mut trace = OcrTrace {
            selected_source: Some("det".to_string()),
            det_pass_count: 1,
            fallback_attempt_count: 0,
            rec_primary_call_count: 1,
            rec_alt_call_count: 0,
            timing: OcrTraceTiming::default(),
            lines: ocr_trace_lines_from_regions(&regions),
            candidates: vec![OcrTraceCandidate {
                label: "det".to_string(),
                mode: "single".to_string(),
                action: "adopted".to_string(),
                reason: "adopted".to_string(),
                score: 92.0,
                confidence: 0.82,
                char_count: 13,
                line_count: 1,
                region_count: 1,
                source_family_count: 1,
            }],
            json: None,
        };

        let json = ocr_trace_json(100, 50, false, 1, 0, false, false, 0.82, &trace, &regions);
        trace.json = Some(json.clone());

        assert!(json.contains("\"width\":100"));
        assert!(json.contains("\"source\":\"det\""));
        assert!(json.contains("\"crop_size\":[80,24]"));
        assert!(json.contains("\"avg_margin\":0.1100"));
        assert!(json.contains("\"min_margin\":0.0500"));
        assert!(json.contains("\"candidate_count\":1"));
        assert!(json.contains("\"adopted_candidate_count\":1"));
        assert!(json.contains("\"source_count\":1"));
        assert!(json.contains("\"candidates\":["));
        assert!(json.contains("\"char_min_confidence\":0.8200"));
        assert!(json.contains("\"support_count\":1"));
        assert!(json.contains("\"label\":\"det\""));
        assert!(json.contains("\"reason\":\"adopted\""));
        assert!(json.contains("A \\\"quoted\\\" line"));
        assert_eq!(trace.json.as_deref(), Some(json.as_str()));
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
    fn tight_rec_crop_removes_blank_margin() {
        let mut rgb = image::RgbImage::from_pixel(200, 48, image::Rgb([255, 255, 255]));
        for y in 20..26 {
            for x in 80..120 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let cropped = tight_rec_crop(&DynamicImage::ImageRgb8(rgb)).expect("tight crop");
        let (w, h) = cropped.dimensions();

        assert!(w < 70);
        assert!(h < 20);
    }

    #[test]
    fn select_recognition_can_choose_alt_for_ascii_line() {
        let primary = RecCandidate {
            text: "川川川".to_string(),
            confidence: 0.61,
            variant: RecVariant::Primary,
            avg_margin: 0.02,
            min_margin: 0.01,
            char_min_confidence: 0.61,
        };
        let alt = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.60,
            variant: RecVariant::Alt,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: 0.60,
        };
        let chosen = select_recognition(primary, Some(alt));
        assert_eq!(chosen.variant, RecVariant::Alt);
        assert_eq!(chosen.text, "Invoice 42");
    }

    #[test]
    fn ctc_decode_reports_character_margins() {
        let alphabet = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let logits = vec![
            0.05, 0.70, 0.05, // blank
            0.90, 0.10, 0.10, // A
            0.20, 0.05, 0.80, // B
            0.10, 0.05, 0.10, // C
        ];

        let (text, confidence, stats) =
            ctc_greedy_decode_with_stats(&logits, &[1, 4, 3], &alphabet);

        assert_eq!(text, "AB");
        assert!((0.60..=1.0).contains(&confidence));
        assert!(stats.avg_margin > 0.40);
        assert!(stats.min_margin > 0.35);
    }

    #[test]
    fn ctc_decode_calibrates_raw_logits_to_probability_confidence() {
        let alphabet = vec!["A".to_string(), "B".to_string()];
        let logits = vec![
            0.0, 0.0, // blank
            4.0, 0.2, // A
            0.1, 3.5, // B
        ];

        let (text, confidence, stats) =
            ctc_greedy_decode_with_stats(&logits, &[1, 3, 2], &alphabet);

        assert_eq!(text, "AB");
        assert!((0.0..=1.0).contains(&confidence));
        assert!((0.0..=1.0).contains(&stats.avg_margin));
        assert!(stats.avg_margin > 0.80);
    }

    #[test]
    fn select_recognition_prefers_primary_when_alt_is_not_clear_win() {
        let primary = RecCandidate {
            text: "测试文本".to_string(),
            confidence: 0.68,
            variant: RecVariant::Primary,
            avg_margin: 0.08,
            min_margin: 0.04,
            char_min_confidence: 0.68,
        };
        let alt = RecCandidate {
            text: "Test Text".to_string(),
            confidence: 0.60,
            variant: RecVariant::Alt,
            avg_margin: 0.08,
            min_margin: 0.04,
            char_min_confidence: 0.60,
        };
        let chosen = select_recognition(primary.clone(), Some(alt));
        assert_eq!(chosen.variant, primary.variant);
        assert_eq!(chosen.text, primary.text);
    }

    #[test]
    fn select_recognition_uses_margin_for_model_quality() {
        let primary = rec_candidate("Status", 0.62, RecVariant::Primary);
        let mut alt = rec_candidate("Status", 0.61, RecVariant::Alt);
        alt.avg_margin = 0.85;
        alt.min_margin = 0.50;

        let chosen = select_recognition(primary, Some(alt));

        assert_eq!(chosen.variant, RecVariant::Alt);
    }

    #[test]
    fn alt_recognition_is_skipped_for_strong_non_ascii_primary() {
        let mut primary = rec_candidate("测试文本", 0.84, RecVariant::Primary);
        primary.avg_margin = 0.12;
        primary.min_margin = 0.08;

        assert!(!should_try_alt_recognition(&primary));
    }

    #[test]
    fn alt_recognition_is_skipped_for_stable_short_candidate() {
        let mut primary = rec_candidate("Alpha", 0.93, RecVariant::Primary);
        primary.avg_margin = 0.12;
        primary.min_margin = 0.08;

        assert!(!should_try_alt_recognition(&primary));
    }

    #[test]
    fn alt_recognition_runs_for_ascii_like_primary() {
        let mut primary = rec_candidate("Invoice 42", 0.71, RecVariant::Primary);
        primary.avg_margin = 0.10;
        primary.min_margin = 0.05;

        assert!(should_try_alt_recognition(&primary));
    }

    #[test]
    fn usable_recognition_rejects_low_quality_text() {
        let repeated = RecCandidate {
            text: "||||||".to_string(),
            confidence: 0.91,
            variant: RecVariant::Primary,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: 0.91,
        };
        let low_confidence = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.12,
            variant: RecVariant::Alt,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: 0.12,
        };
        let valid = RecCandidate {
            text: "Invoice 42".to_string(),
            confidence: 0.42,
            variant: RecVariant::Alt,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: 0.42,
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
    fn quality_fallback_budget_is_tighter_for_partial_text() {
        assert_eq!(
            quality_fallback_family_budget(""),
            MAX_QUALITY_FALLBACK_FAMILIES_EMPTY
        );
        assert_eq!(
            quality_fallback_family_budget("partial text"),
            MAX_QUALITY_FALLBACK_FAMILIES_PARTIAL
        );
    }

    #[test]
    fn normalize_recognized_text_splits_joined_chat_time_marker() {
        assert_eq!(
            normalize_recognized_text("Sender: alpha build刚刚next update"),
            "Sender: alpha build\n刚刚next update"
        );
        assert_eq!(
            normalize_recognized_text("plain message without marker"),
            "plain message without marker"
        );
    }

    #[test]
    fn candidate_text_lines_split_newlines_into_separate_boxes() {
        let mut rgb = image::RgbImage::from_pixel(120, 48, image::Rgb([255, 255, 255]));
        for y in 8..14 {
            for x in 12..78 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        for y in 30..36 {
            for x in 10..90 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let candidate = RecCandidate {
            text: "Alpha\nBeta".to_string(),
            confidence: 0.82,
            variant: RecVariant::Primary,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: 0.82,
        };

        let lines = candidate_text_lines(
            &DynamicImage::ImageRgb8(rgb),
            (0, 0, 120, 48),
            &candidate,
            "det",
            BboxTransform::Identity,
        );

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Alpha");
        assert_eq!(lines[1].text, "Beta");
        assert_eq!(lines[0].source, "det:multiline");
        assert!(lines[0].bbox.3 <= lines[1].bbox.1);
    }

    #[test]
    fn recognized_box_repair_flags_low_quality_or_large_weak_lines() {
        assert!(recognized_box_needs_repair((0, 0, 100, 24), "Alpha", 0.30));
        assert!(recognized_box_needs_repair(
            (0, 0, 620, 90),
            "Alpha Beta",
            0.70
        ));
        assert!(recognized_box_needs_repair((0, 0, 140, 24), "||||||", 0.90));
        assert!(!recognized_box_needs_repair(
            (0, 0, 160, 28),
            "Alpha Beta",
            0.86
        ));
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
    fn candidate_pool_merges_lines_from_multiple_paths() {
        let mut first_lines = vec![text_line((10, 10, 100, 26), "Alpha", 0.72)];
        let first = recognized_from_text_lines(&mut first_lines);
        let mut second_lines = vec![text_line((10, 44, 110, 60), "Beta", 0.70)];
        second_lines[0].source = "layered-region:eager".to_string();
        let second = recognized_from_text_lines(&mut second_lines);
        let pool = vec![
            OcrCandidateEntry {
                label: "det".to_string(),
                recognized: first,
            },
            OcrCandidateEntry {
                label: "layered-regions:eager".to_string(),
                recognized: second,
            },
        ];

        let recognized = recognized_from_candidate_pool(&pool);

        assert_eq!(recognized.line_count, 2);
        assert!(recognized.text.contains("Alpha"));
        assert!(recognized.text.contains("Beta"));
    }

    #[test]
    fn recognized_from_text_lines_tracks_support_count_for_duplicate_votes() {
        let mut lines = vec![
            text_line((10, 10, 190, 26), "Project status: ready", 0.62),
            text_line((11, 10, 191, 26), "Project status: ready", 0.61),
            text_line((12, 11, 192, 27), "Project status: reedy", 0.74),
        ];
        lines[1].source = "tile-region:1".to_string();

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 1);
        assert!(recognized.regions[0].lines[0].support_count >= 2);
    }

    #[test]
    fn maybe_adopt_candidate_pool_adopts_extra_supported_line() {
        let mut text = "Alpha".to_string();
        let mut confidence = 0.72;
        let mut line_count = 1;
        let mut region_count = 1;
        let mut layout_applied = false;
        let mut regions = vec![OcrTextRegion {
            bbox: [10, 10, 100, 26],
            text: "Alpha".to_string(),
            confidence: 0.72,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 100, 26],
                text: "Alpha".to_string(),
                confidence: 0.72,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.72,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];
        let mut fallback = None;
        let mut pool = vec![OcrCandidateEntry {
            label: "det".to_string(),
            recognized: RecognizedText {
                text: "Alpha".to_string(),
                confidence: 0.72,
                line_count: 1,
                region_count: 1,
                layout_applied: false,
                regions: regions.clone(),
            },
        }];
        let mut candidate_lines = vec![text_line((10, 44, 110, 60), "Beta", 0.68)];
        candidate_lines[0].source = "color-region:eager".to_string();
        let candidate = recognized_from_text_lines(&mut candidate_lines);

        assert!(maybe_adopt_candidate_pool(
            &mut text,
            &mut confidence,
            &mut line_count,
            &mut region_count,
            &mut layout_applied,
            &mut regions,
            &mut fallback,
            &mut pool,
            None,
            "color-regions:eager".to_string(),
            &candidate,
        ));
        assert_eq!(line_count, 2);
        assert!(text.contains("Alpha"));
        assert!(text.contains("Beta"));
        assert_eq!(fallback.as_deref(), Some("pooled:color-regions:eager"));
    }

    #[test]
    fn foreground_binary_variants_include_mask_fusions() {
        let mut rgb = image::RgbImage::from_pixel(96, 32, image::Rgb([236, 242, 248]));
        draw_synthetic_text_texture_at(&mut rgb, 10, 80, 10, 4);

        let variants = foreground_binary_variants(&DynamicImage::ImageRgb8(rgb));
        let names = variants
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        assert!(names.contains(&"foreground-binary"));
        assert!(names.iter().any(|name| name.contains("foreground-")));
        assert!(!variants.is_empty());
    }

    #[test]
    fn panel_child_candidate_boxes_keep_nested_panels() {
        let mut rgb = image::RgbImage::from_pixel(320, 180, image::Rgb([250, 250, 250]));
        fill_rect(&mut rgb, (20, 20, 140, 80), image::Rgb([232, 239, 248]));
        fill_rect(&mut rgb, (170, 24, 300, 90), image::Rgb([236, 245, 233]));
        draw_synthetic_text_texture_at(&mut rgb, 34, 120, 34, 4);
        draw_synthetic_text_texture_at(&mut rgb, 184, 286, 40, 4);

        let boxes = panel_child_candidate_boxes(&DynamicImage::ImageRgb8(rgb));

        assert!(boxes.len() >= 2);
        assert!(boxes.iter().any(|b| b.0 < 60 && b.2 < 180));
        assert!(boxes.iter().any(|b| b.0 > 140 && b.2 > 260));
    }

    #[test]
    fn low_threshold_box_thresh_is_bounded() {
        assert!((low_threshold_box_thresh(0.20) - 0.144).abs() < 1.0e-6);
        assert_eq!(low_threshold_box_thresh(0.30), 0.18);
        assert_eq!(low_threshold_box_thresh(0.05), 0.10);
    }

    #[test]
    fn traced_single_candidate_records_rejected_reason() {
        let mut text = "Alpha Beta".to_string();
        let mut confidence = 0.86;
        let mut line_count = 1;
        let mut region_count = 1;
        let mut layout_applied = false;
        let mut regions = vec![ocr_region_with_line(
            [10, 10, 160, 30],
            "Alpha Beta",
            0.86,
            "det",
        )];
        let mut fallback = None;
        let mut trace = OcrTrace::default();
        let candidate = RecognizedText {
            text: "Alpha".to_string(),
            confidence: 0.60,
            line_count: 1,
            region_count: 1,
            layout_applied: false,
            regions: vec![ocr_region_with_line(
                [10, 10, 80, 30],
                "Alpha",
                0.60,
                "visual-region",
            )],
        };

        assert!(!maybe_adopt_recognized_traced(
            &mut text,
            &mut confidence,
            &mut line_count,
            &mut region_count,
            &mut layout_applied,
            &mut regions,
            &mut fallback,
            &mut trace,
            "visual-regions:eager".to_string(),
            &candidate,
        ));

        assert_eq!(trace.candidates.len(), 1);
        assert_eq!(trace.candidates[0].action, "rejected");
        assert_eq!(trace.candidates[0].reason, "not-better");
    }

    #[test]
    fn recognized_from_text_lines_dedupes_overlapping_similar_lines() {
        let mut lines = vec![
            text_line((10, 10, 180, 26), "Project status: ready", 0.62),
            text_line((12, 11, 182, 27), "Project status: reedy", 0.70),
            text_line((10, 90, 180, 106), "Release notes: ready", 0.63),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 2);
        assert!(recognized.text.contains("Project status: reedy"));
        assert!(recognized.text.contains("Release notes: ready"));
    }

    #[test]
    fn recognized_from_text_lines_votes_across_near_duplicate_sources() {
        let mut lines = vec![
            text_line((10, 10, 190, 26), "Project status: reedy", 0.74),
            text_line((11, 10, 191, 26), "Project status: ready", 0.62),
            text_line((12, 11, 192, 27), "Project status: ready", 0.61),
        ];
        lines[1].source = "tile-region:1".to_string();
        lines[2].source = "color-region-det:eager:1".to_string();

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 1);
        assert!(recognized.text.contains("Project status: ready"));
        assert!(!recognized.text.contains("Project status: reedy"));
    }

    #[test]
    fn recognized_from_text_lines_dedupes_nearby_long_text_variants() {
        let mut lines = vec![
            text_line((320, 10, 540, 26), "Project Alpha Release 2024", 0.70),
            text_line((322, 54, 538, 70), "Project Alfa Release 2024", 0.74),
            text_line((320, 300, 520, 316), "Quarterly Sales Summary", 0.68),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 2);
    }

    #[test]
    fn recognized_from_text_lines_dedupes_distant_same_column_long_variants() {
        let mut lines = vec![
            text_line((320, 10, 540, 26), "Project Alpha Release 2024", 0.70),
            text_line((322, 360, 538, 376), "Project Alfa Release 2024", 0.74),
            text_line((24, 360, 130, 376), "Toolbar", 0.80),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 2);
        assert!(recognized.text.contains("Toolbar"));
    }

    #[test]
    fn recognized_from_text_lines_dedupes_global_exact_long_text_only() {
        let mut lines = vec![
            text_line((20, 10, 260, 26), "Project Alpha Release 2024", 0.70),
            text_line((360, 420, 600, 436), "Project Alpha Release 2024", 0.76),
            text_line((20, 60, 80, 76), "短名", 0.70),
            text_line((360, 460, 420, 476), "短名", 0.76),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(
            recognized
                .text
                .matches("Project Alpha Release 2024")
                .count(),
            1
        );
        assert_eq!(recognized.text.matches("短名").count(), 2);
    }

    #[test]
    fn recognized_from_text_lines_drops_short_ascii_noise_in_dense_results() {
        let mut lines = vec![
            text_line((0, 0, 20, 12), "AM", 0.92),
            text_line((0, 20, 20, 32), "+", 0.88),
            text_line((0, 40, 20, 52), "WH", 0.90),
            text_line((40, 0, 130, 12), "Q搜索", 0.81),
            text_line((40, 20, 130, 32), "工作台", 0.82),
            text_line((40, 40, 130, 52), "发送(S)", 0.83),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.line_count, 3);
        assert!(!recognized.text.contains("AM"));
        assert!(!recognized.text.contains("WH"));
        assert!(!recognized.text.contains("+"));
        assert!(recognized.text.contains("Q搜索"));
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
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.78,
                readable_ratio: 1.0,
                support_count: 1,
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
    fn uncovered_color_region_filter_allows_repairable_overlap() {
        let reliable = vec![OcrTextRegion {
            bbox: [10, 10, 100, 32],
            text: "Alpha".to_string(),
            confidence: 0.86,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 100, 32],
                text: "Alpha".to_string(),
                confidence: 0.86,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: 0.86,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];
        let weak = vec![OcrTextRegion {
            bbox: [10, 10, 100, 32],
            text: "||||||".to_string(),
            confidence: 0.82,
            source: "det".to_string(),
            lines: vec![OcrTextLine {
                bbox: [10, 10, 100, 32],
                text: "||||||".to_string(),
                confidence: 0.82,
                avg_margin: 0.01,
                min_margin: 0.0,
                char_min_confidence: 0.82,
                readable_ratio: 1.0,
                support_count: 1,
                source: "det".to_string(),
            }],
        }];

        assert!(color_region_box_covered_by_reliable_text(
            (12, 12, 98, 30),
            &reliable
        ));
        assert!(!color_region_box_covered_by_reliable_text(
            (12, 12, 98, 30),
            &weak
        ));
    }

    #[test]
    fn supplement_candidate_priority_prefers_dense_text_like_box() {
        let mut rgb = image::RgbImage::from_pixel(240, 120, image::Rgb([250, 250, 250]));
        for y in 20..24 {
            for x in 10..210 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        for y in 64..79 {
            for x in 20..120 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = prioritize_supplement_candidate_boxes(
            &DynamicImage::ImageRgb8(rgb),
            vec![(0, 10, 230, 50), (0, 55, 140, 90)],
            &[],
        );

        assert_eq!(boxes[0], (0, 55, 140, 90));
        assert_eq!(boxes[1], (0, 10, 230, 50));
    }

    #[test]
    fn primary_content_focus_band_prefers_wide_centered_regions() {
        let rgb = image::RgbImage::from_pixel(480, 220, image::Rgb([248, 248, 248]));
        let existing = vec![
            OcrTextRegion {
                bbox: [148, 24, 344, 38],
                text: "主内容1".to_string(),
                confidence: 0.82,
                source: "det".to_string(),
                lines: vec![OcrTextLine {
                    bbox: [148, 24, 344, 38],
                    text: "主内容1".to_string(),
                    confidence: 0.82,
                    avg_margin: 0.0,
                    min_margin: 0.0,
                    char_min_confidence: 0.82,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                }],
            },
            OcrTextRegion {
                bbox: [144, 68, 340, 82],
                text: "主内容2".to_string(),
                confidence: 0.80,
                source: "det".to_string(),
                lines: vec![OcrTextLine {
                    bbox: [144, 68, 340, 82],
                    text: "主内容2".to_string(),
                    confidence: 0.80,
                    avg_margin: 0.0,
                    min_margin: 0.0,
                    char_min_confidence: 0.80,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                }],
            },
            OcrTextRegion {
                bbox: [152, 112, 348, 126],
                text: "主内容3".to_string(),
                confidence: 0.81,
                source: "det".to_string(),
                lines: vec![OcrTextLine {
                    bbox: [152, 112, 348, 126],
                    text: "主内容3".to_string(),
                    confidence: 0.81,
                    avg_margin: 0.0,
                    min_margin: 0.0,
                    char_min_confidence: 0.81,
                    readable_ratio: 1.0,
                    support_count: 1,
                    source: "det".to_string(),
                }],
            },
        ];

        let focus_band =
            primary_content_focus_band(&DynamicImage::ImageRgb8(rgb), &existing).unwrap();
        assert!(focus_band.0 <= 120);
        assert!(focus_band.1 >= 370);
        assert!(box_intersects_focus_band(
            (132, 18, 360, 42),
            focus_band.0,
            focus_band.1
        ));
        assert!(!box_intersects_focus_band(
            (0, 18, 96, 42),
            focus_band.0,
            focus_band.1
        ));
    }

    #[test]
    fn supplement_candidate_priority_caps_outside_focus_candidates() {
        let focus_band = Some((120, 420));
        let mut scored = vec![
            ((0, 16, 96, 40), 140),
            ((160, 16, 408, 40), 120),
            ((0, 62, 98, 86), 138),
            ((158, 62, 406, 86), 118),
            ((0, 108, 100, 132), 136),
            ((166, 108, 414, 132), 116),
            ((0, 154, 96, 178), 134),
            ((160, 154, 408, 178), 114),
            ((444, 20, 540, 44), 132),
            ((442, 66, 538, 90), 130),
            ((448, 112, 544, 136), 128),
            ((440, 158, 536, 182), 126),
        ];
        sort_supplement_candidate_scores(&mut scored, focus_band);
        let boxes = retain_focus_prioritized_candidates(scored, focus_band)
            .into_iter()
            .map(|(b, _)| b)
            .collect::<Vec<_>>();

        let outside = boxes.iter().filter(|b| b.2 <= 120 || b.0 >= 430).count();
        assert!(outside <= MAX_SUPPLEMENT_OUTSIDE_FOCUS_CANDIDATES);
        assert!(boxes[..4].iter().all(|b| b.0 >= 150 && b.2 <= 420));
    }

    #[test]
    fn supplement_box_is_redundant_when_overlap_is_high() {
        let seen = vec![
            SupplementSeenBox {
                bbox: (16, 20, 188, 44),
                has_reliable_text: true,
            },
            SupplementSeenBox {
                bbox: (240, 24, 324, 48),
                has_reliable_text: true,
            },
        ];
        assert!(supplement_box_is_redundant((18, 21, 186, 43), &seen));
        assert!(!supplement_box_is_redundant((10, 18, 36, 36), &seen));
        assert!(!supplement_box_is_redundant((220, 12, 300, 40), &seen));
    }

    #[test]
    fn supplement_box_is_not_redundant_until_reliable() {
        let seen = vec![SupplementSeenBox {
            bbox: (16, 20, 188, 44),
            has_reliable_text: false,
        }];
        assert!(!supplement_box_is_redundant((18, 21, 186, 43), &seen));
    }

    #[test]
    fn supplement_candidate_priority_prefers_focus_boxes_over_higher_scored_sidebar_boxes() {
        let focus_band = Some((120, 420));
        let mut scored = vec![
            ((0, 16, 96, 40), 180),
            ((160, 16, 408, 40), 120),
            ((444, 20, 540, 44), 170),
            ((158, 62, 406, 86), 118),
            ((166, 108, 414, 132), 116),
        ];
        sort_supplement_candidate_scores(&mut scored, focus_band);
        let boxes = scored.into_iter().map(|(b, _)| b).collect::<Vec<_>>();

        assert_eq!(boxes[0], (160, 16, 408, 40));
        assert_eq!(boxes[1], (158, 62, 406, 86));
        assert_eq!(boxes[2], (166, 108, 414, 132));
        assert_eq!(boxes[3], (0, 16, 96, 40));
        assert_eq!(boxes[4], (444, 20, 540, 44));
    }

    #[test]
    fn layered_color_region_text_boxes_extract_panel_foreground_lines() {
        let mut rgb = image::RgbImage::from_pixel(260, 150, image::Rgb([248, 248, 248]));
        for y in 24..126 {
            for x in 30..230 {
                rgb.put_pixel(x, y, image::Rgb([232, 235, 240]));
            }
        }
        for y in 50..58 {
            for x in 54..178 {
                rgb.put_pixel(x, y, image::Rgb([24, 24, 24]));
            }
        }
        for y in 88..96 {
            for x in 58..196 {
                rgb.put_pixel(x, y, image::Rgb([24, 24, 24]));
            }
        }

        let boxes = layered_color_region_text_boxes(&DynamicImage::ImageRgb8(rgb), &[]);

        assert!(boxes.len() >= 2);
        assert!(boxes.iter().any(|b| b.1 <= 50 && b.3 >= 58));
        assert!(boxes.iter().any(|b| b.1 <= 88 && b.3 >= 96));
        assert!(boxes.iter().all(|b| box_height(*b) < 60));
    }

    #[test]
    fn dominant_color_layer_boxes_keep_subtle_dark_foreground() {
        let mut rgb = image::RgbImage::from_pixel(180, 70, image::Rgb([236, 238, 242]));
        for y in 24..32 {
            for x in [24, 44, 64, 104, 124, 144] {
                for xx in x..x + 8 {
                    rgb.put_pixel(xx, y, image::Rgb([208, 210, 214]));
                }
            }
        }

        let boxes = dominant_color_layer_text_boxes(&DynamicImage::ImageRgb8(rgb), 4);

        assert!(!boxes.is_empty());
        assert!(boxes.iter().any(|b| b.1 <= 24 && b.3 >= 32));
    }

    #[test]
    fn local_recognition_variants_include_multi_window_binaries() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 24, Luma([220])));
        let names = local_recognition_variants(&img)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"local-small".to_string()));
        assert!(names.contains(&"local-medium".to_string()));
        assert!(names.contains(&"local-large-invert".to_string()));
    }

    #[test]
    fn foreground_binary_variants_limited_respects_budget() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 24, Luma([220])));
        let variants = foreground_binary_variants_limited(&img, 1);
        assert!(variants.len() <= 1);
    }

    #[test]
    fn local_recognition_variants_limited_respects_budget() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 24, Luma([220])));
        let variants = local_recognition_variants_limited(&img, 1);
        assert_eq!(variants.len(), 1);
        assert!(local_recognition_variants_limited(&img, 2).len() <= 2);
    }

    #[test]
    fn local_recognition_variants_skip_when_direct_is_stable() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(120, 32, Luma([255])));
        let mut stable = rec_candidate("Stable text", 0.86, RecVariant::Primary);
        stable.avg_margin = 0.18;
        stable.min_margin = 0.08;
        let weak = rec_candidate("Stable text", 0.52, RecVariant::Primary);

        assert!(!should_try_local_recognition_variants(&img, Some(&stable)));
        assert!(should_try_local_recognition_variants(&img, Some(&weak)));
    }

    #[test]
    fn local_recognition_variants_adaptive_limits_medium_direct_result() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(120, 32, Luma([220])));
        let candidate = rec_candidate("Panel text", 0.62, RecVariant::Primary);

        let variants = local_recognition_variants_adaptive(&img, Some(&candidate));

        assert!(variants.len() <= 3);
        assert!(!variants.is_empty());
    }

    #[test]
    fn local_recognition_variants_adaptive_stops_for_tiny_crop_short_text() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(40, 30, Luma([220])));
        let candidate = rec_candidate("AB", 0.72, RecVariant::Primary);
        let variants = local_recognition_variants_adaptive(&img, Some(&candidate));

        assert_eq!(variants.len(), 1);
    }

    #[test]
    fn local_recognition_candidate_is_final_returns_false_for_weak_candidates() {
        let mut candidate = rec_candidate("测试", 0.91, RecVariant::Primary);
        assert!(!local_recognition_candidate_is_final(&candidate));

        candidate.confidence = 0.95;
        candidate.avg_margin = 0.16;
        candidate.min_margin = 0.10;
        candidate.char_min_confidence = 0.96;
        assert!(local_recognition_candidate_is_final(&candidate));
    }

    #[test]
    fn quality_fallback_enhancement_variant_budget_scales_with_confidence() {
        let weak = rec_candidate("AB", 0.36, RecVariant::Primary);
        let weak_limit =
            quality_fallback_enhancement_variant_budget(&weak.text, weak.confidence, 8, 1);
        assert_eq!(weak_limit, MAX_QUALITY_FALLBACK_ENHANCEMENT_VARIANTS);

        let strong_limit = quality_fallback_enhancement_variant_budget("标题", 0.89, 2, 2);
        assert_eq!(strong_limit, MAX_ENHANCEMENT_VARIANTS_PER_PASS);
    }

    #[test]
    fn crop_enhancement_variants_skip_for_strong_direct_candidate() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(220, 40, Luma([220])));
        let mut candidate = rec_candidate("AB", 0.93, RecVariant::Primary);
        candidate.avg_margin = 0.15;
        candidate.min_margin = 0.10;
        assert!(!should_try_crop_enhancement_variants(Some(&candidate)));
        assert_eq!(crop_enhancement_variant_budget(&img, Some(&candidate)), 0);
    }

    #[test]
    fn crop_enhancement_variants_keep_for_uncertain_direct_candidate() {
        let candidate = rec_candidate("Unstable text", 0.72, RecVariant::Primary);
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(360, 100, Luma([220])));
        assert!(should_try_crop_enhancement_variants(Some(&candidate)));
        assert_eq!(crop_enhancement_variant_budget(&img, Some(&candidate)), 4);
    }

    #[test]
    fn crop_enhancement_variants_limit_small_crops_even_when_uncertain() {
        let candidate = rec_candidate("Maybe", 0.72, RecVariant::Primary);
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(60, 16, Luma([220])));
        assert!(should_try_crop_enhancement_variants(Some(&candidate)));
        assert_eq!(crop_enhancement_variant_budget(&img, Some(&candidate)), 1);
    }

    #[test]
    fn crop_enhancement_variants_limit_short_text_in_small_box() {
        let candidate = rec_candidate("短字", 0.72, RecVariant::Primary);
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(75, 80, Luma([220])));
        assert_eq!(crop_enhancement_variant_budget(&img, Some(&candidate)), 1);
    }

    #[test]
    fn crop_enhancement_variants_skip_strong_short_candidate() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(220, 40, Luma([220])));
        let mut candidate = rec_candidate("AB", 0.93, RecVariant::Primary);
        candidate.avg_margin = 0.15;
        candidate.min_margin = 0.10;
        assert!(!should_try_crop_enhancement_variants(Some(&candidate)));
        assert_eq!(crop_enhancement_variant_budget(&img, Some(&candidate)), 0);
    }

    #[test]
    fn crop_enhancement_candidate_is_final_requires_high_quality() {
        let mut candidate = rec_candidate("AB", 0.98, RecVariant::Primary);
        candidate.avg_margin = 0.16;
        candidate.min_margin = 0.06;
        assert!(crop_enhancement_candidate_is_final(&candidate));

        candidate.confidence = 0.95;
        assert!(!crop_enhancement_candidate_is_final(&candidate));
    }

    #[test]
    fn rec_candidate_cache_key_distinguishes_targets_and_variant() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(140, 40, Luma([128])));
        let cfg = OcrConfig::default();
        let primary = rec_candidate_cache_key(&img, &cfg, 120, RecVariant::Primary);
        let primary_fallback = rec_candidate_cache_key(&img, &cfg, 240, RecVariant::Primary);
        let alt = rec_candidate_cache_key(&img, &cfg, 120, RecVariant::Alt);

        assert_ne!(primary, primary_fallback);
        assert_ne!(primary, alt);
    }

    #[test]
    fn rec_candidate_cache_reuses_inserted_candidate() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 20, Luma([128])));
        let cfg = OcrConfig::default();
        let key = rec_candidate_cache_key(&img, &cfg, 96, RecVariant::Primary);
        let candidate = rec_candidate("测试", 0.91, RecVariant::Primary);

        let mut cache = RecCandidateCache::default();
        cache.put(key, candidate.clone());

        assert_eq!(cache.get(&key).map(|it| it.text.as_str()), Some("测试"));
        assert_eq!(cache.get(&key).map(|it| it.confidence), Some(0.91));
    }

    #[test]
    fn local_det_upscale_variants_prioritize_two_x_when_budget_allows() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(120, 40, Luma([128])));
        let variants = local_det_upscale_variants(&img)
            .into_iter()
            .map(|(name, image)| (name, image.dimensions()))
            .collect::<Vec<_>>();

        assert_eq!(variants[0], ("local-upscaled-2x".to_string(), (240, 80)));
    }

    #[test]
    fn scale_offset_transform_maps_local_upscaled_box_to_page() {
        let mapped = BboxTransform::ScaleOffset {
            sx: 0.5,
            sy: 0.5,
            dx: 100,
            dy: 40,
            max_w: 400,
            max_h: 240,
        }
        .map_box((20, 10, 80, 30));

        assert_eq!(mapped, (110, 45, 140, 55));
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
    fn layout_regions_respect_horizontal_visual_separator() {
        let mut rgb = image::RgbImage::from_pixel(160, 80, image::Rgb([255, 255, 255]));
        for x in 10..150 {
            rgb.put_pixel(x, 32, image::Rgb([60, 60, 60]));
        }
        let image = DynamicImage::ImageRgb8(rgb);
        let mut without_context = vec![
            text_line((20, 10, 120, 22), "Alpha", 0.82),
            text_line((20, 44, 120, 56), "Beta", 0.82),
        ];
        let mut with_context = without_context.clone();

        let merged = recognized_from_text_lines(&mut without_context);
        let separated = recognized_from_text_lines_with_image(&mut with_context, &image);

        assert_eq!(merged.region_count, 1);
        assert_eq!(separated.region_count, 2);
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
    fn layout_graph_merges_same_row_but_not_cross_column_header() {
        let mut lines = vec![
            text_line((0, 0, 320, 14), "Header", 0.90),
            text_line((0, 44, 70, 58), "Left", 0.82),
            text_line((86, 44, 150, 58), "Item", 0.81),
            text_line((210, 44, 310, 58), "Right", 0.83),
        ];

        let recognized = recognized_from_text_lines(&mut lines);

        assert_eq!(recognized.text, "Header\n\nLeft\nItem\n\nRight");
        assert_eq!(recognized.region_count, 3);
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
    fn color_region_binarization_handles_low_contrast_dark_text() {
        let mut rgb = image::RgbImage::from_pixel(96, 32, image::Rgb([240, 240, 240]));
        for y in 13..18 {
            for x in 24..72 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }

        let binary =
            binarize_color_region_foreground(&DynamicImage::ImageRgb8(rgb), (0, 0, 96, 32))
                .expect("low contrast binary region");
        let gray = binary.to_luma8();

        assert_eq!(gray.get_pixel(4, 4)[0], 255);
        assert_eq!(gray.get_pixel(36, 15)[0], 0);
    }

    #[test]
    fn color_region_binarization_rejects_flat_low_contrast_region() {
        let rgb = image::RgbImage::from_pixel(96, 32, image::Rgb([240, 240, 240]));

        let binary =
            binarize_color_region_foreground(&DynamicImage::ImageRgb8(rgb), (0, 0, 96, 32));

        assert!(binary.is_none());
    }

    #[test]
    fn foreground_line_boxes_use_low_contrast_mask() {
        let mut rgb = image::RgbImage::from_pixel(140, 72, image::Rgb([240, 240, 240]));
        for y in 14..20 {
            for x in 18..112 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }
        for y in 46..52 {
            for x in 24..120 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }

        let boxes = foreground_line_boxes(&DynamicImage::ImageRgb8(rgb), 4);

        assert_eq!(boxes.len(), 2);
        assert!(boxes[0].1 <= 14 && boxes[0].3 >= 20);
        assert!(boxes[1].1 <= 46 && boxes[1].3 >= 52);
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
    fn forced_structural_split_boxes_split_large_low_contrast_panel() {
        let mut rgb = image::RgbImage::from_pixel(320, 140, image::Rgb([240, 240, 240]));
        for y in 24..31 {
            for x in 28..250 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }
        for y in 66..73 {
            for x in 32..280 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }
        for y in 106..113 {
            for x in 30..210 {
                rgb.put_pixel(x, y, image::Rgb([228, 228, 228]));
            }
        }

        let boxes =
            forced_structural_split_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 320, 140), 6);

        assert_eq!(boxes.len(), 3);
        assert!(boxes[0].1 <= 24 && boxes[0].3 >= 31);
        assert!(boxes[1].1 <= 66 && boxes[1].3 >= 73);
        assert!(boxes[2].1 <= 106 && boxes[2].3 >= 113);
    }

    #[test]
    fn forced_structural_split_skips_regular_single_line_box() {
        assert!(!large_text_box_needs_structured_split((0, 0, 640, 35)));
        assert!(large_text_box_needs_structured_split((0, 0, 640, 140)));
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
    fn wide_line_segment_boxes_split_long_line_at_narrow_gaps() {
        let mut rgb = image::RgbImage::from_pixel(660, 48, image::Rgb([255, 255, 255]));
        draw_segment_texture(&mut rgb, 20, 210);
        draw_segment_texture(&mut rgb, 235, 425);
        draw_segment_texture(&mut rgb, 450, 640);

        let boxes = wide_line_segment_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 660, 48), 4);

        assert_eq!(boxes.len(), 3);
        assert!(boxes.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn wide_line_segment_limit_grows_for_very_long_lines() {
        assert_eq!(wide_line_segment_limit((0, 0, 640, 48), 8), 4);
        assert_eq!(wide_line_segment_limit((0, 0, 960, 40), 8), 5);
        assert_eq!(wide_line_segment_limit((0, 0, 1400, 42), 8), 6);
    }

    #[test]
    fn wide_line_segment_boxes_avoid_hard_cut_without_gap() {
        let mut rgb = image::RgbImage::from_pixel(660, 48, image::Rgb([255, 255, 255]));
        for y in 14..32 {
            for x in 20..640 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }

        let boxes = wide_line_segment_boxes(&DynamicImage::ImageRgb8(rgb), (0, 0, 660, 48), 4);

        assert!(boxes.is_empty());
    }

    #[test]
    fn join_segment_recognition_text_spaces_ascii_only() {
        let ascii = vec![
            rec_candidate("Alpha", 0.80, RecVariant::Primary),
            rec_candidate("Beta", 0.82, RecVariant::Primary),
        ];
        let cjk = vec![
            rec_candidate("甲方", 0.80, RecVariant::Primary),
            rec_candidate("乙方", 0.82, RecVariant::Primary),
        ];

        assert_eq!(join_segment_recognition_text(&ascii), "Alpha Beta");
        assert_eq!(join_segment_recognition_text(&cjk), "甲方乙方");
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
    fn deskew_estimates_small_foreground_angle() {
        let mut rgb = image::RgbImage::from_pixel(180, 80, image::Rgb([255, 255, 255]));
        for x in 20..160 {
            let y = 30 + (x - 20) / 16;
            for dy in 0..3 {
                rgb.put_pixel(x, y + dy, image::Rgb([20, 20, 20]));
            }
        }

        let angle = estimate_foreground_skew_degrees(&DynamicImage::ImageRgb8(rgb))
            .expect("estimated angle");

        assert!(angle > 2.0);
        assert!(angle < 7.0);
    }

    #[test]
    fn rotate_image_degrees_on_white_expands_canvas() {
        let img = DynamicImage::ImageLuma8(GrayImage::from_pixel(80, 30, Luma([255])));
        let rotated = rotate_image_degrees_on_white(&img, 5.0);

        assert!(rotated.width() > 80);
        assert!(rotated.height() >= 30);
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

    #[test]
    fn voted_lines_use_margin_for_near_duplicate_quality() {
        let mut weak_margin = text_line((10, 10, 180, 26), "Project status: ready", 0.73);
        weak_margin.avg_margin = 0.01;
        weak_margin.min_margin = 0.0;
        let mut strong_margin = text_line((11, 10, 181, 26), "Project status: ready", 0.70);
        strong_margin.avg_margin = 0.80;
        strong_margin.min_margin = 0.45;

        let selected = select_voted_text_line(&[weak_margin, strong_margin]).expect("line");

        assert_eq!(selected.confidence, 0.70);
        assert!(selected.avg_margin > 0.50);
    }

    #[test]
    fn low_value_ascii_noise_requires_internal_mixed_case_or_short_token() {
        assert!(is_low_value_ascii_noise("abCdef"));
        assert!(is_low_value_ascii_noise("fox"));
        assert!(!is_low_value_ascii_noise("Invoice"));
        assert!(!is_low_value_ascii_noise("Alpha42"));
    }

    #[test]
    fn usable_recognition_rejects_low_confidence_ascii_noise() {
        let noisy = rec_candidate("abCdef", 0.65, RecVariant::Primary);
        let ordinary = rec_candidate("Invoice", 0.65, RecVariant::Primary);

        assert!(!is_usable_recognition(&noisy));
        assert!(is_usable_recognition(&ordinary));
    }

    #[test]
    fn large_boxes_can_prioritize_structural_split() {
        assert!(large_text_box_should_prioritize_split((0, 0, 900, 120)));
        assert!(large_text_box_should_prioritize_split((0, 0, 520, 420)));
        assert!(!large_text_box_should_prioritize_split((0, 0, 640, 35)));
    }

    #[test]
    fn structured_split_lines_require_readable_content() {
        let good = vec![
            text_line((0, 10, 220, 30), "Alpha row", 0.62),
            text_line((0, 54, 240, 74), "Beta row", 0.64),
        ];
        let bad = vec![
            text_line((0, 10, 20, 18), "x", 0.92),
            text_line((0, 54, 20, 62), "+", 0.94),
        ];

        assert!(structured_split_lines_are_plausible(
            (0, 0, 320, 120),
            &good
        ));
        assert!(!structured_split_lines_are_plausible(
            (0, 0, 320, 120),
            &bad
        ));
    }

    #[test]
    fn wide_line_sliding_window_boxes_split_continuous_long_line() {
        let mut rgb = image::RgbImage::from_pixel(900, 48, image::Rgb([255, 255, 255]));
        for y in 14..32 {
            for x in 20..880 {
                rgb.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
        let img = DynamicImage::ImageRgb8(rgb);

        assert!(wide_line_segment_boxes(&img, (0, 0, 900, 48), 4).is_empty());
        let windows = wide_line_recognition_boxes(&img, (0, 0, 900, 48), 4);

        assert!(windows.len() >= 2);
        assert!(windows.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(windows.windows(2).any(|pair| pair[0].2 > pair[1].0));
    }

    #[test]
    fn segment_join_dedupes_overlap_between_windows() {
        let segments = vec![
            rec_candidate("AlphaBeta", 0.80, RecVariant::Primary),
            rec_candidate("BetaGamma", 0.81, RecVariant::Primary),
        ];

        assert_eq!(join_segment_recognition_text(&segments), "AlphaBetaGamma");
    }

    #[test]
    fn best_panel_for_line_uses_overlap_and_center() {
        let panels = vec![(0, 0, 300, 200), (300, 0, 640, 200)];

        assert_eq!(best_panel_for_line((40, 20, 160, 40), &panels), Some(0));
        assert_eq!(best_panel_for_line((420, 20, 560, 40), &panels), Some(1));
    }

    #[test]
    fn foreground_component_boxes_merge_text_like_components() {
        let mut rgb = image::RgbImage::from_pixel(220, 80, image::Rgb([245, 245, 245]));
        for x in [24, 48, 72, 120, 146, 172] {
            for y in 28..42 {
                for xx in x..x + 10 {
                    rgb.put_pixel(xx, y, image::Rgb([20, 20, 20]));
                }
            }
        }

        let boxes = foreground_component_text_boxes(&DynamicImage::ImageRgb8(rgb), 4);

        assert!(!boxes.is_empty());
        assert!(boxes.iter().any(|b| box_width(*b) > 120));
    }

    #[test]
    fn glyph_textness_penalizes_solid_icon_but_keeps_text_texture() {
        let mut solid = vec![false; 32 * 32];
        for y in 8..24 {
            for x in 8..24 {
                solid[y * 32 + x] = true;
            }
        }
        let mut text = vec![false; 96 * 24];
        for x in [8, 18, 28, 48, 58, 68] {
            for y in 8..16 {
                for xx in x..x + 5 {
                    text[y * 96 + xx] = true;
                }
            }
        }

        let solid_score = foreground_glyph_textness_score(&solid, 32, 32).expect("solid score");
        let text_score = foreground_glyph_textness_score(&text, 96, 24).expect("text score");

        assert!(solid_score < 0);
        assert!(text_score > solid_score);
    }

    #[test]
    fn ctc_path_beam_decode_produces_candidate_with_stats() {
        let alphabet = vec!["A".to_string(), "B".to_string()];
        let logits = vec![
            0.05, 0.70, 0.05, // blank
            0.90, 0.10, 0.10, // A
            0.20, 0.05, 0.80, // B
        ];

        let (text, confidence, stats) =
            ctc_path_beam_decode_with_stats(&logits, &[1, 3, 3], &alphabet).expect("beam");

        assert_eq!(text, "AB");
        assert!(confidence > 0.70);
        assert!(stats.avg_margin > 0.60);
    }

    #[test]
    fn ctc_prefix_beam_aggregates_same_text_probability() {
        let alphabet = vec!["A".to_string(), "B".to_string()];
        let logits = vec![
            0.05, 0.45, 0.50, // step 1
            0.46, 0.45, 0.09, // step 2
        ];

        let greedy = ctc_greedy_decode_with_stats(&logits, &[1, 2, 3], &alphabet);
        let beam = ctc_path_beam_decode_with_stats(&logits, &[1, 2, 3], &alphabet).expect("beam");

        assert_eq!(greedy.0, "B");
        assert_eq!(beam.0, "A");
    }

    fn text_line(bbox: BoxRect, text: &str, confidence: f32) -> TextLine {
        make_text_line(
            bbox,
            text.to_string(),
            confidence,
            0.10,
            0.06,
            "det".to_string(),
        )
    }

    fn rec_candidate(text: &str, confidence: f32, variant: RecVariant) -> RecCandidate {
        RecCandidate {
            text: text.to_string(),
            confidence,
            variant,
            avg_margin: 0.10,
            min_margin: 0.06,
            char_min_confidence: confidence,
        }
    }

    fn ocr_region_with_line(
        bbox: [u32; 4],
        text: &str,
        confidence: f32,
        source: &str,
    ) -> OcrTextRegion {
        OcrTextRegion {
            bbox,
            text: text.to_string(),
            confidence,
            source: source.to_string(),
            lines: vec![OcrTextLine {
                bbox,
                text: text.to_string(),
                confidence,
                avg_margin: 0.10,
                min_margin: 0.06,
                char_min_confidence: confidence,
                readable_ratio: readable_ratio(text),
                support_count: 1,
                source: source.to_string(),
            }],
        }
    }

    fn detection_box(bbox: BoxRect) -> DetectionBox {
        DetectionBox {
            bbox,
            alternatives: Vec::new(),
        }
    }

    fn fill_rect(rgb: &mut image::RgbImage, b: BoxRect, color: image::Rgb<u8>) {
        for y in b.1..b.3.min(rgb.height()) {
            for x in b.0..b.2.min(rgb.width()) {
                rgb.put_pixel(x, y, color);
            }
        }
    }

    fn draw_synthetic_text_texture(rgb: &mut image::RgbImage, x0: u32, x1: u32) {
        draw_synthetic_text_texture_at(rgb, x0, x1, 24, 10);
    }

    fn draw_synthetic_text_texture_at(
        rgb: &mut image::RgbImage,
        x0: u32,
        x1: u32,
        y0: u32,
        rows: u32,
    ) {
        let dark = image::Rgb([24, 24, 24]);
        for row in 0..rows {
            let y = y0 + row * 18;
            let mut x = x0 + (row % 3) * 5;
            while x + 16 < x1 {
                for yy in y..(y + 5).min(rgb.height()) {
                    for xx in x..(x + 16).min(rgb.width()) {
                        rgb.put_pixel(xx, yy, dark);
                    }
                }
                x += 28;
            }
        }
    }

    fn draw_segment_texture(rgb: &mut image::RgbImage, x0: u32, x1: u32) {
        let dark = image::Rgb([20, 20, 20]);
        let mut x = x0;
        while x + 10 < x1 {
            for y in 14..32 {
                for xx in x..(x + 12).min(x1).min(rgb.width()) {
                    rgb.put_pixel(xx, y, dark);
                }
            }
            x += 18;
        }
    }
}
