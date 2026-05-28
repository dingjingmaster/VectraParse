pub mod content_handler;

use vectraparse_core::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParseOutcome {
    pub content: Option<String>,
    pub metadata: Metadata,
    pub warnings: Vec<String>,
    pub parser_chain: Vec<String>,
}

pub trait Parser: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, media_type: &str) -> bool;
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome>;
}

pub struct CompositeParser {
    parsers: Vec<Box<dyn Parser>>,
}

impl CompositeParser {
    pub fn new(parsers: Vec<Box<dyn Parser>>) -> Self {
        Self { parsers }
    }

    pub fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        let mut supplement = ParseOutcome::default();
        for parser in &self.parsers {
            if !parser.supports(media_type) {
                continue;
            }
            if let Some(mut out) = parser.parse(input, media_type) {
                out.parser_chain.insert(0, parser.name().to_string());
                if out.content.is_some() {
                    let mut warnings = std::mem::take(&mut out.warnings);
                    supplement.warnings.append(&mut warnings);
                    for k in out.metadata.keys().map(ToString::to_string).collect::<Vec<_>>() {
                        if let Some(vals) = out.metadata.values(&k) {
                            for v in vals {
                                supplement.metadata.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    if out.content.is_some() {
                        out.warnings = supplement.warnings.clone();
                        for k in supplement
                            .metadata
                            .keys()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                        {
                            if let Some(vals) = supplement.metadata.values(&k) {
                                for v in vals {
                                    out.metadata.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                    return Some(out);
                }
                supplement.warnings.extend(out.warnings);
                for k in out.metadata.keys().map(ToString::to_string).collect::<Vec<_>>() {
                    if let Some(vals) = out.metadata.values(&k) {
                        for v in vals {
                            supplement.metadata.insert(k.clone(), v.clone());
                        }
                    }
                }
                supplement.parser_chain.extend(out.parser_chain);
            }
        }
        None
    }

    pub fn parse_multiple(&self, input: &[u8], media_type: &str) -> Vec<ParseOutcome> {
        self.parsers
            .iter()
            .filter(|p| p.supports(media_type))
            .filter_map(|p| {
                p.parse(input, media_type).map(|mut out| {
                    out.parser_chain.insert(0, p.name().to_string());
                    out
                })
            })
            .collect()
    }
}

pub struct TextParser;
impl Parser for TextParser {
    fn name(&self) -> &'static str {
        "TextParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/plain" || media_type == "text/csv"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let mut metadata = Metadata::default();
        metadata.insert("parser", "TextParser");
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct HtmlParser;
impl Parser for HtmlParser {
    fn name(&self) -> &'static str {
        "HtmlParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/html"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let mut metadata = Metadata::default();
        metadata.insert("parser", "HtmlParser");
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct MetadataOnlyParser;
impl Parser for MetadataOnlyParser {
    fn name(&self) -> &'static str {
        "MetadataOnlyParser"
    }
    fn supports(&self, _media_type: &str) -> bool {
        true
    }
    fn parse(&self, _input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let mut metadata = Metadata::default();
        metadata.insert("supplement", "true");
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings: vec!["supplement-parser".to_string()],
            parser_chain: vec!["MetadataOnlyParser".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CompositeParser, HtmlParser, MetadataOnlyParser, TextParser};

    #[test]
    fn mime_to_parser_mapping_and_fallback() {
        let composite = CompositeParser::new(vec![
            Box::new(MetadataOnlyParser),
            Box::new(HtmlParser),
            Box::new(TextParser),
        ]);
        let text = composite
            .parse(b"hello", "text/plain")
            .expect("text parser should parse");
        assert_eq!(text.content.as_deref(), Some("hello"));
        assert!(text.parser_chain.contains(&"TextParser".to_string()));
        assert!(text.metadata.values("supplement").is_some());
        assert!(composite.parse(b"\xFF\xFE", "application/pdf").is_none());
    }

    #[test]
    fn multiple_parser_dispatch_returns_all_matches() {
        let composite = CompositeParser::new(vec![Box::new(TextParser), Box::new(MetadataOnlyParser)]);
        let all = composite.parse_multiple(b"a,b,c", "text/csv");
        assert_eq!(all.len(), 2);
    }
}
