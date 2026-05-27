#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorProvider {
    pub name: String,
    pub priority: i32,
    pub feature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserProvider {
    pub name: String,
    pub media_types: Vec<String>,
    pub feature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProviderRegistry {
    detectors: Vec<DetectorProvider>,
    parsers: Vec<ParserProvider>,
}

impl ProviderRegistry {
    pub fn with_defaults() -> Self {
        let mut reg = Self::default();
        reg.register_detector("OverrideDetector", 1000, None);
        reg.register_detector("POIFSContainerDetector", 900, None);
        reg.register_detector("ZipContainerDetector", 800, None);
        reg.register_detector("BPListDetector", 700, None);
        reg.register_detector("NameDetector", 100, None);
        reg.register_detector("TypeDetector", 50, None);

        reg.register_parser("TextAndCSVParser", &["text/plain", "text/csv"], None);
        reg.register_parser("HtmlParser", &["text/html"], None);
        reg.register_parser("PDFParser", &["application/pdf"], None);
        reg.register_parser(
            "TesseractOCRParser",
            &["image/tiff", "image/png", "image/jpeg"],
            Some("ocr"),
        );
        reg
    }

    pub fn register_detector(
        &mut self,
        name: &str,
        priority: i32,
        feature: Option<&str>,
    ) {
        self.detectors.push(DetectorProvider {
            name: name.to_string(),
            priority,
            feature: feature.map(ToString::to_string),
        });
        self.detectors
            .sort_by(|a, b| b.priority.cmp(&a.priority).then(a.name.cmp(&b.name)));
    }

    pub fn register_parser(&mut self, name: &str, media_types: &[&str], feature: Option<&str>) {
        self.parsers.push(ParserProvider {
            name: name.to_string(),
            media_types: media_types.iter().map(|m| (*m).to_string()).collect(),
            feature: feature.map(ToString::to_string),
        });
        self.parsers.sort_by(|a, b| a.name.cmp(&b.name));
    }

    pub fn detector_names(&self) -> Vec<String> {
        self.detectors.iter().map(|d| d.name.clone()).collect()
    }

    pub fn parser_names(&self) -> Vec<String> {
        self.parsers.iter().map(|p| p.name.clone()).collect()
    }

    pub fn parsers_for_media_type(&self, media_type: &str) -> Vec<String> {
        self.parsers
            .iter()
            .filter(|p| p.media_types.iter().any(|m| m == media_type))
            .map(|p| p.name.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderRegistry;

    #[test]
    fn provider_snapshot_order() {
        let reg = ProviderRegistry::with_defaults();
        assert_eq!(
            reg.detector_names(),
            vec![
                "OverrideDetector",
                "POIFSContainerDetector",
                "ZipContainerDetector",
                "BPListDetector",
                "NameDetector",
                "TypeDetector"
            ]
        );
        assert_eq!(
            reg.parser_names(),
            vec![
                "HtmlParser",
                "PDFParser",
                "TesseractOCRParser",
                "TextAndCSVParser"
            ]
        );
    }

    #[test]
    fn query_parser_by_media_type() {
        let reg = ProviderRegistry::with_defaults();
        assert_eq!(
            reg.parsers_for_media_type("text/csv"),
            vec!["TextAndCSVParser".to_string()]
        );
    }
}
