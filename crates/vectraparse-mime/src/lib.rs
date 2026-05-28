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
    use super::{MediaTypeRegistry, generated_stats, source_path};

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
}
