use crate::{ParseOutcome, Parser};
use vectraparse_core::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationConfig {
    pub fork_isolation_enabled: bool,
    pub network_parser_enabled: bool,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            fork_isolation_enabled: false,
            network_parser_enabled: false,
        }
    }
}

pub struct DigestHashParser;
impl Parser for DigestHashParser {
    fn name(&self) -> &'static str {
        "DigestHashParser"
    }
    fn supports(&self, _media_type: &str) -> bool {
        true
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let sum: u64 = input.iter().fold(0u64, |acc, b| acc.wrapping_add(*b as u64));
        let mut md = Metadata::default();
        md.insert("digest.sum64", format!("{sum:016x}"));
        Some(ParseOutcome {
            content: None,
            metadata: md,
            warnings: Vec::new(),
            parser_chain: vec!["DigestHashParser".to_string()],
        })
    }
}

pub struct NetworkParser {
    pub config: IsolationConfig,
}

impl Parser for NetworkParser {
    fn name(&self) -> &'static str {
        "NetworkParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/plain"
    }
    fn parse(&self, _input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if !self.config.network_parser_enabled {
            let mut md = Metadata::default();
            md.insert("network.disabled", "true");
            return Some(ParseOutcome {
                content: None,
                metadata: md,
                warnings: vec!["network-parser-disabled".to_string()],
                parser_chain: vec!["NetworkParser".to_string()],
            });
        }
        let mut md = Metadata::default();
        md.insert("network.enabled", "true");
        let mut warnings = Vec::new();
        if !self.config.fork_isolation_enabled {
            warnings.push("isolation-disabled".to_string());
        }
        Some(ParseOutcome {
            content: Some("network-parser-result".to_string()),
            metadata: md,
            warnings,
            parser_chain: vec!["NetworkParser".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{DigestHashParser, IsolationConfig, NetworkParser};
    use crate::Parser;

    #[test]
    fn digest_hash_parser_outputs_hash_metadata() {
        let parser = DigestHashParser;
        let out = parser.parse(b"abc", "text/plain").expect("hash output");
        assert_eq!(
            out.metadata
                .values("digest.sum64")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("0000000000000126")
        );
    }

    #[test]
    fn network_parser_can_be_disabled() {
        let p = NetworkParser {
            config: IsolationConfig::default(),
        };
        let out = p.parse(b"x", "text/plain").expect("out");
        assert!(out.warnings.iter().any(|w| w == "network-parser-disabled"));
    }
}
