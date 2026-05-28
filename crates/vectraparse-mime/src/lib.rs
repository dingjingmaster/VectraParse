use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimeStats {
    pub mime_type_count: usize,
    pub magic_count: usize,
    pub glob_count: usize,
    pub root_xml_count: usize,
    pub alias_count: usize,
    pub subclass_count: usize,
}

mod generated {
    include!(concat!(env!("OUT_DIR"), "/mime_generated.rs"));
}

pub fn generated_stats() -> MimeStats {
    MimeStats {
        mime_type_count: generated::MIME_TYPE_COUNT,
        magic_count: generated::MAGIC_COUNT,
        glob_count: generated::GLOB_COUNT,
        root_xml_count: generated::ROOT_XML_COUNT,
        alias_count: generated::ALIAS_COUNT,
        subclass_count: generated::SUBCLASS_COUNT,
    }
}

pub fn source_path() -> &'static str {
    generated::TIKA_MIME_XML_PATH
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MagicRule {
    pub mime: &'static str,
    pub priority: i32,
    pub offset: usize,
    pub pattern: &'static [u8],
    pub mask: Option<&'static [u8]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MagicMatch {
    pub mime: String,
    pub priority: i32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetectHints<'a> {
    pub resource_name: Option<&'a str>,
    pub content_type_hint: Option<&'a str>,
    pub force_content_type: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorConfig {
    pub enable_override_detector: bool,
    pub enable_zero_size_detector: bool,
    pub enable_name_detector: bool,
    pub enable_type_detector: bool,
    pub enable_trained_model_detector: bool,
    pub enable_nn_example_detector: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            enable_override_detector: true,
            enable_zero_size_detector: true,
            enable_name_detector: true,
            enable_type_detector: true,
            enable_trained_model_detector: false,
            enable_nn_example_detector: false,
        }
    }
}

pub fn detector_provider_names(cfg: &DetectorConfig) -> Vec<&'static str> {
    let mut out = Vec::new();
    if cfg.enable_override_detector {
        out.push("OverrideDetector");
    }
    if cfg.enable_zero_size_detector {
        out.push("ZeroSizeFileDetector");
    }
    if cfg.enable_name_detector {
        out.push("NameDetector");
    }
    if cfg.enable_type_detector {
        out.push("TypeDetector");
    }
    if cfg.enable_trained_model_detector {
        out.push("TrainedModelDetector");
    }
    if cfg.enable_nn_example_detector {
        out.push("NNExampleModelDetector");
    }
    out
}

#[derive(Debug, Default)]
pub struct MagicMatcher {
    rules: Vec<MagicRule>,
}

impl MagicMatcher {
    pub fn from_rules(mut rules: Vec<MagicRule>) -> Self {
        rules.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.offset.cmp(&b.offset))
                .then(a.mime.cmp(b.mime))
        });
        Self { rules }
    }

    pub fn default_rules() -> Self {
        Self::from_rules(vec![
            MagicRule {
                mime: "application/pdf",
                priority: 80,
                offset: 0,
                pattern: b"%PDF-",
                mask: None,
            },
            MagicRule {
                mime: "application/zip",
                priority: 50,
                offset: 0,
                pattern: b"PK\x03\x04",
                mask: None,
            },
            MagicRule {
                mime: "application/zip",
                priority: 50,
                offset: 0,
                pattern: b"PK\x05\x06",
                mask: None,
            },
            MagicRule {
                mime: "application/zip",
                priority: 50,
                offset: 0,
                pattern: b"PK\x07\x08",
                mask: None,
            },
            MagicRule {
                mime: "application/x-tika-msoffice",
                priority: 45,
                offset: 0,
                pattern: b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1",
                mask: None,
            },
            MagicRule {
                mime: "application/x-bplist",
                priority: 44,
                offset: 0,
                pattern: b"bplist00",
                mask: None,
            },
            MagicRule {
                mime: "application/octet-stream",
                priority: 1,
                offset: 0,
                pattern: b"\x00\x00\x00\x00",
                mask: Some(b"\x00\x00\x00\x00"),
            },
        ])
    }

    pub fn detect(&self, input: &[u8], read_window: usize) -> MagicMatch {
        if input.is_empty() {
            return MagicMatch {
                mime: "application/x-empty".to_string(),
                priority: i32::MAX,
            };
        }
        let window = input.len().min(read_window);
        let sliced = &input[..window];
        for rule in &self.rules {
            if magic_matches_rule(sliced, rule) {
                return MagicMatch {
                    mime: rule.mime.to_string(),
                    priority: rule.priority,
                };
            }
        }
        MagicMatch {
            mime: "application/octet-stream".to_string(),
            priority: i32::MIN,
        }
    }
}

fn magic_matches_rule(input: &[u8], rule: &MagicRule) -> bool {
    let end = match rule.offset.checked_add(rule.pattern.len()) {
        Some(v) => v,
        None => return false,
    };
    if end > input.len() {
        return false;
    }
    let actual = &input[rule.offset..end];
    match rule.mask {
        None => actual == rule.pattern,
        Some(mask) => {
            if mask.len() != rule.pattern.len() {
                return false;
            }
            actual
                .iter()
                .zip(rule.pattern.iter())
                .zip(mask.iter())
                .all(|((a, expected), m)| (a & m) == (expected & m))
        }
    }
}

pub fn detect_media_type(input: &[u8], hints: &DetectHints<'_>) -> String {
    detect_media_type_with_config(input, hints, &DetectorConfig::default())
}

pub fn detect_media_type_with_config(
    input: &[u8],
    hints: &DetectHints<'_>,
    cfg: &DetectorConfig,
) -> String {
    let registry = MediaTypeRegistry::from_generated();
    if cfg.enable_override_detector {
        if let Some(forced) = hints
            .force_content_type
            .and_then(|v| registry.normalize(v).filter(|m| registry.knows(m)))
        {
            return forced;
        }
    }
    if cfg.enable_type_detector {
        if let Some(hint) = hints
            .content_type_hint
            .and_then(|v| registry.normalize(v).filter(|m| registry.knows(m)))
        {
            return hint;
        }
    }
    if cfg.enable_zero_size_detector && input.is_empty() {
        return "application/x-empty".to_string();
    }
    if cfg.enable_name_detector {
        if let Some(by_name) = hints
            .resource_name
            .and_then(media_type_from_resource_name)
            .and_then(|v| registry.normalize(v).filter(|m| registry.knows(m)))
        {
            return by_name;
        }
    }
    if (cfg.enable_trained_model_detector || cfg.enable_nn_example_detector)
        && looks_like_model_hint(input)
    {
        return "application/x-model-predicted".to_string();
    }
    let magic = MagicMatcher::default_rules().detect(input, 64).mime;
    if magic == "application/zip" {
        return specialize_zip_container(input).to_string();
    }
    if magic == "application/x-tika-msoffice" {
        return specialize_ole_container(input).to_string();
    }
    if magic == "application/x-bplist" {
        return "application/x-bplist".to_string();
    }
    if magic == "application/octet-stream" {
        if let Some(plist) = detect_apple_xml_plist(input) {
            return plist.to_string();
        }
        if let Some(refined) = detect_xml_html_or_text(input) {
            return refined;
        }
    }
    magic
}

fn looks_like_model_hint(input: &[u8]) -> bool {
    let probe = String::from_utf8_lossy(&input[..input.len().min(4096)]).to_ascii_lowercase();
    probe.contains("neural-model:") || probe.contains("ml-prediction:")
}

fn media_type_from_resource_name(name: &str) -> Option<&'static str> {
    let trimmed = name.trim();
    let (_, ext) = trimmed.rsplit_once('.')?;
    let ext = ext.trim().to_ascii_lowercase();
    match ext.as_str() {
        "pdf" => Some("application/pdf"),
        "zip" => Some("application/zip"),
        "txt" => Some("text/plain"),
        "htm" | "html" => Some("text/html"),
        "xml" => Some("application/xml"),
        "csv" => Some("text/csv"),
        "json" => Some("application/json"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        _ => None,
    }
}

fn specialize_zip_container(input: &[u8]) -> &'static str {
    let probe = String::from_utf8_lossy(&input[..input.len().min(128 * 1024)]);
    if probe.contains("word/") || probe.contains("ppt/") || probe.contains("xl/") {
        return "application/x-tika-ooxml";
    }
    if probe.contains("mimetypeapplication/vnd.oasis.opendocument.text")
        || probe.contains("mimetypeapplication/vnd.oasis.opendocument.spreadsheet")
        || probe.contains("mimetypeapplication/vnd.oasis.opendocument.presentation")
    {
        return "application/vnd.oasis.opendocument";
    }
    if probe.contains("META-INF/container.xml") || probe.contains("OEBPS/") {
        return "application/epub+zip";
    }
    if probe.contains("Index/Document.iwa")
        || probe.contains("Index/Tables/")
        || probe.contains("Index/Metadata.iwa")
    {
        return "application/x-iwork-package";
    }
    if probe.contains("AndroidManifest.xml") || probe.contains("classes.dex") {
        return "application/vnd.android.package-archive";
    }
    if probe.contains("META-INF/MANIFEST.MF") {
        return "application/java-archive";
    }
    "application/zip"
}

fn specialize_ole_container(input: &[u8]) -> &'static str {
    let probe = String::from_utf8_lossy(&input[..input.len().min(512 * 1024)]);
    if probe.contains("WordDocument") {
        return "application/msword";
    }
    if probe.contains("Workbook") || probe.contains("Book") {
        return "application/vnd.ms-excel";
    }
    if probe.contains("PowerPoint Document") {
        return "application/vnd.ms-powerpoint";
    }
    if probe.contains("__properties_version1.0") || probe.contains("__substg1.0_") {
        return "application/vnd.ms-outlook";
    }
    if probe.contains("MSysObjects") || probe.contains("Standard Jet DB") {
        return "application/x-msaccess";
    }
    if probe.contains("~$") || probe.contains("OwnerFile") {
        return "application/x-tika-msoffice";
    }
    "application/x-tika-msoffice"
}

fn detect_xml_html_or_text(input: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(input).ok()?.trim_start_matches('\u{feff}').trim_start();
    if s.is_empty() {
        return None;
    }
    if s.starts_with("<!DOCTYPE html")
        || s.starts_with("<html")
        || s.starts_with("<HTML")
        || s.contains("<meta charset=")
    {
        return Some("text/html".to_string());
    }
    if s.starts_with("<?xml") || s.starts_with('<') {
        if let Some(root) = first_xml_root(s) {
            let root_lower = root.to_ascii_lowercase();
            if root_lower == "html" {
                return Some("text/html".to_string());
            }
            if root_lower == "feed" {
                return Some("application/atom+xml".to_string());
            }
            if root_lower == "rss" {
                return Some("application/rss+xml".to_string());
            }
            return Some("application/xml".to_string());
        }
    }
    if looks_like_plain_text(input) {
        return Some("text/plain".to_string());
    }
    None
}

fn first_xml_root(s: &str) -> Option<&str> {
    let mut in_tag = false;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if !in_tag && b == b'<' {
            if i + 1 < bytes.len() && (bytes[i + 1] == b'?' || bytes[i + 1] == b'!') {
                i += 1;
            } else {
                in_tag = true;
                start = i + 1;
            }
        } else if in_tag {
            if b == b'>' || b.is_ascii_whitespace() || b == b'/' {
                return s.get(start..i);
            }
        }
        i += 1;
    }
    None
}

fn looks_like_plain_text(input: &[u8]) -> bool {
    if std::str::from_utf8(input).is_ok() {
        let mut printable = 0usize;
        let mut total = 0usize;
        for &b in input.iter().take(4096) {
            total += 1;
            if b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7e).contains(&b) {
                printable += 1;
            }
        }
        if total == 0 {
            return false;
        }
        return printable * 100 / total >= 90;
    }
    false
}

fn detect_apple_xml_plist(input: &[u8]) -> Option<&'static str> {
    let s = std::str::from_utf8(input).ok()?;
    let t = s.trim_start_matches('\u{feff}').trim_start();
    if t.starts_with("<?xml")
        && t.contains("<!DOCTYPE plist")
        && t.contains("<plist")
        && t.contains("</plist>")
    {
        return Some("application/x-plist");
    }
    None
}

#[derive(Debug, Default)]
pub struct MediaTypeRegistry {
    aliases: HashMap<&'static str, &'static str>,
    superclasses: HashMap<&'static str, Vec<&'static str>>,
    known: HashSet<&'static str>,
}

impl MediaTypeRegistry {
    pub fn from_generated() -> Self {
        let mut aliases = HashMap::new();
        for (alias, canonical) in generated::ALIAS_PAIRS {
            aliases.insert(*alias, *canonical);
        }
        let mut superclasses = HashMap::new();
        for (child, parent) in generated::SUBCLASS_PAIRS {
            superclasses
                .entry(*child)
                .or_insert_with(Vec::new)
                .push(*parent);
        }
        let known = generated::KNOWN_MIME_TYPES.iter().copied().collect();
        Self {
            aliases,
            superclasses,
            known,
        }
    }

    pub fn normalize(&self, raw: &str) -> Option<String> {
        let mut current = normalize_media_type(raw)?;
        let mut guard = 0usize;
        while let Some(canonical) = self.aliases.get(current.as_str()) {
            if *canonical == current {
                break;
            }
            current = (*canonical).to_string();
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        Some(current)
    }

    pub fn supertypes(&self, raw: &str) -> Vec<String> {
        let Some(normalized) = self.normalize(raw) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut stack = vec![normalized];
        let mut seen = HashSet::new();
        while let Some(mt) = stack.pop() {
            if !seen.insert(mt.clone()) {
                continue;
            }
            if let Some(next) = self.superclasses.get(mt.as_str()) {
                for parent in next {
                    out.push((*parent).to_string());
                    stack.push((*parent).to_string());
                }
            }
        }
        out
    }

    pub fn direct_supertype(&self, raw: &str) -> Option<String> {
        let normalized = self.normalize(raw)?;
        self.superclasses
            .get(normalized.as_str())
            .and_then(|v| v.first().copied())
            .map(ToString::to_string)
    }

    pub fn is_specialization_of(&self, raw: &str, maybe_parent: &str) -> bool {
        let Some(child) = self.normalize(raw) else {
            return false;
        };
        let Some(parent) = self.normalize(maybe_parent) else {
            return false;
        };
        if child == parent {
            return true;
        }
        self.supertypes(&child).iter().any(|p| p == &parent)
    }

    pub fn specialize(&self, parent: &str, candidate: &str) -> Option<String> {
        let normalized_candidate = self.normalize(candidate)?;
        if self.is_specialization_of(&normalized_candidate, parent) {
            return Some(normalized_candidate);
        }
        None
    }

    pub fn knows(&self, raw: &str) -> bool {
        self.normalize(raw)
            .is_some_and(|normalized| self.known.contains(normalized.as_str()))
    }
}

fn normalize_media_type(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let no_params = trimmed.split(';').next()?.trim();
    let (ty, sub) = no_params.split_once('/')?;
    let ty = ty.trim().to_ascii_lowercase();
    let sub = sub.trim().to_ascii_lowercase();
    if ty.is_empty() || sub.is_empty() {
        return None;
    }
    Some(format!("{ty}/{sub}"))
}

#[cfg(test)]
mod tests {
    use super::{
        DetectHints, DetectorConfig, MagicMatcher, MagicRule, MediaTypeRegistry, detect_media_type,
        detector_provider_names, generated_stats, source_path,
    };

    #[test]
    fn generated_counts_match_expected_tika_snapshot() {
        let s = generated_stats();
        assert_eq!(s.mime_type_count, 1599);
        assert_eq!(s.magic_count, 355);
        assert_eq!(s.glob_count, 1302);
        assert_eq!(s.root_xml_count, 62);
        assert_eq!(s.alias_count, 141);
        assert_eq!(s.subclass_count, 321);
    }

    #[test]
    fn source_path_is_stable() {
        assert!(source_path().ends_with("tika-mimetypes.xml"));
    }

    #[test]
    fn registry_normalize_alias_and_params() {
        let reg = MediaTypeRegistry::from_generated();
        assert_eq!(
            reg.normalize(" Application/X-Javascript ; charset=utf-8 ")
                .as_deref(),
            Some("application/javascript")
        );
    }

    #[test]
    fn registry_supertype_and_specialize_work() {
        let reg = MediaTypeRegistry::from_generated();
        assert_eq!(
            reg.direct_supertype("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                .as_deref(),
            Some("application/x-tika-ooxml")
        );
        assert!(
            reg.supertypes("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                .iter()
                .any(|v| v == "application/zip")
        );
        assert!(
            reg.is_specialization_of(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/zip"
            )
        );
        assert_eq!(
            reg.specialize(
                "application/zip",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            .as_deref(),
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        );
    }

    #[test]
    fn magic_matcher_detects_empty_file() {
        let matcher = MagicMatcher::default_rules();
        let matched = matcher.detect(&[], 32);
        assert_eq!(matched.mime, "application/x-empty");
    }

    #[test]
    fn magic_matcher_applies_priority_and_offset() {
        let matcher = MagicMatcher::from_rules(vec![
            MagicRule {
                mime: "application/test-low",
                priority: 10,
                offset: 1,
                pattern: b"ABC",
                mask: None,
            },
            MagicRule {
                mime: "application/test-high",
                priority: 100,
                offset: 1,
                pattern: b"ABC",
                mask: None,
            },
        ]);
        let matched = matcher.detect(b"XABCY", 8);
        assert_eq!(matched.mime, "application/test-high");
    }

    #[test]
    fn magic_matcher_supports_mask_and_short_input() {
        let matcher = MagicMatcher::from_rules(vec![MagicRule {
            mime: "application/masked",
            priority: 5,
            offset: 0,
            pattern: b"\xF0\x0A",
            mask: Some(b"\xF0\x0F"),
        }]);
        assert_eq!(matcher.detect(b"\xFA\x0A", 2).mime, "application/masked");
        assert_eq!(
            matcher.detect(b"\xFA", 2).mime,
            "application/octet-stream"
        );
    }

    #[test]
    fn detect_chain_prioritizes_force_then_hint_then_name_then_magic() {
        let bytes = b"%PDF-1.7\n...";
        assert_eq!(
            detect_media_type(
                bytes,
                &DetectHints {
                    force_content_type: Some("application/zip"),
                    content_type_hint: Some("application/pdf"),
                    resource_name: Some("x.docx"),
                }
            ),
            "application/zip"
        );
        assert_eq!(
            detect_media_type(
                bytes,
                &DetectHints {
                    force_content_type: None,
                    content_type_hint: Some("application/xml"),
                    resource_name: Some("x.docx"),
                }
            ),
            "application/xml"
        );
        assert_eq!(
            detect_media_type(
                bytes,
                &DetectHints {
                    force_content_type: None,
                    content_type_hint: None,
                    resource_name: Some("x.docx"),
                }
            ),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
    }

    #[test]
    fn detect_chain_falls_back_to_magic() {
        let bytes = b"%PDF-1.7\n...";
        assert_eq!(
            detect_media_type(
                bytes,
                &DetectHints {
                    resource_name: Some("x.unknown"),
                    content_type_hint: None,
                    force_content_type: None
                }
            ),
            "application/pdf"
        );
    }

    #[test]
    fn detect_xml_html_root_and_text_fallback() {
        assert_eq!(
            detect_media_type(
                br#"<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"#,
                &DetectHints::default()
            ),
            "application/atom+xml"
        );
        assert_eq!(
            detect_media_type(
                br#"<!DOCTYPE html><html><head><meta charset="utf-8"></head></html>"#,
                &DetectHints::default()
            ),
            "text/html"
        );
        assert_eq!(
            detect_media_type(b"just plain ascii text", &DetectHints::default()),
            "text/plain"
        );
    }

    #[test]
    fn detect_zip_specialization_chain() {
        assert_eq!(
            detect_media_type(
                b"PK\x03\x04...word/document.xml...[Content_Types].xml",
                &DetectHints::default()
            ),
            "application/x-tika-ooxml"
        );
        assert_eq!(
            detect_media_type(
                b"PK\x03\x04...META-INF/container.xml...OEBPS/content.opf",
                &DetectHints::default()
            ),
            "application/epub+zip"
        );
        assert_eq!(
            detect_media_type(
                b"PK\x03\x04...META-INF/MANIFEST.MF...",
                &DetectHints::default()
            ),
            "application/java-archive"
        );
        assert_eq!(
            detect_media_type(b"PK\x03\x04...random...", &DetectHints::default()),
            "application/zip"
        );
    }

    #[test]
    fn detect_ole_specialization_chain() {
        assert_eq!(
            detect_media_type(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....WordDocument....",
                &DetectHints::default()
            ),
            "application/msword"
        );
        assert_eq!(
            detect_media_type(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....Workbook....",
                &DetectHints::default()
            ),
            "application/vnd.ms-excel"
        );
        assert_eq!(
            detect_media_type(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....__substg1.0_....",
                &DetectHints::default()
            ),
            "application/vnd.ms-outlook"
        );
    }

    #[test]
    fn detect_bplist_and_xml_plist() {
        assert_eq!(
            detect_media_type(b"bplist00....", &DetectHints::default()),
            "application/x-bplist"
        );
        assert_eq!(
            detect_media_type(
                br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict/></plist>"#,
                &DetectHints::default()
            ),
            "application/x-plist"
        );
    }

    #[test]
    fn advanced_detector_providers_are_configurable() {
        let cfg = DetectorConfig::default();
        assert_eq!(
            detector_provider_names(&cfg),
            vec![
                "OverrideDetector",
                "ZeroSizeFileDetector",
                "NameDetector",
                "TypeDetector"
            ]
        );
        let cfg2 = DetectorConfig {
            enable_trained_model_detector: true,
            enable_nn_example_detector: true,
            ..DetectorConfig::default()
        };
        assert!(
            detector_provider_names(&cfg2)
                .contains(&"TrainedModelDetector")
        );
        assert!(
            detector_provider_names(&cfg2)
                .contains(&"NNExampleModelDetector")
        );
    }

    #[test]
    fn model_detector_can_be_enabled_and_disabled() {
        let hints = DetectHints::default();
        let input = b"neural-model:invoice-classifier";
        let disabled = DetectorConfig::default();
        assert_ne!(
            super::detect_media_type_with_config(input, &hints, &disabled),
            "application/x-model-predicted"
        );
        let enabled = DetectorConfig {
            enable_trained_model_detector: true,
            ..DetectorConfig::default()
        };
        assert_eq!(
            super::detect_media_type_with_config(input, &hints, &enabled),
            "application/x-model-predicted"
        );
    }
}
