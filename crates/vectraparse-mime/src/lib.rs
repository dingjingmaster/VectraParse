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
    use super::{MagicMatcher, MagicRule, MediaTypeRegistry, generated_stats, source_path};

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
}
