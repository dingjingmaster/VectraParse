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

#[cfg(test)]
mod tests {
    use super::{generated_stats, source_path};

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
}
