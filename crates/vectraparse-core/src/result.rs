use crate::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParseTiming {
    pub detect_ms: u32,
    pub parse_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbeddedResult {
    pub path: String,
    pub mime_type: String,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StructuredResult {
    pub mime_type: String,
    pub content: Option<String>,
    pub metadata: Metadata,
    pub embedded: Vec<EmbeddedResult>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub parser_chain: Vec<String>,
    pub timing: ParseTiming,
}

impl StructuredResult {
    pub fn to_json(&self) -> String {
        let content = self
            .content
            .as_ref()
            .map(|v| format!("\"{}\"", escape_json(v)))
            .unwrap_or_else(|| "null".to_string());
        let warnings = json_str_array(&self.warnings);
        let errors = json_str_array(&self.errors);
        let parser_chain = json_str_array(&self.parser_chain);
        let embedded = self
            .embedded
            .iter()
            .map(|e| {
                format!(
                    "{{\"path\":\"{}\",\"mime_type\":\"{}\",\"metadata\":{}}}",
                    escape_json(&e.path),
                    escape_json(&e.mime_type),
                    e.metadata.to_json()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"mime_type\":\"{}\",\"metadata\":{},\"content\":{},\"embedded\":[{}],\"warnings\":{},\"errors\":{},\"parser_chain\":{},\"timing\":{{\"detect_ms\":{},\"parse_ms\":{}}}}}",
            escape_json(&self.mime_type),
            self.metadata.to_json(),
            content,
            embedded,
            warnings,
            errors,
            parser_chain,
            self.timing.detect_ms,
            self.timing.parse_ms
        )
    }

    pub fn from_json(input: &str) -> Option<Self> {
        let mime = extract_json_string(input, "mime_type")?;
        let content = extract_json_nullable_string(input, "content")?;
        let warnings = extract_json_string_array(input, "warnings")?;
        let errors = extract_json_string_array(input, "errors")?;
        let parser_chain = extract_json_string_array(input, "parser_chain")?;
        let detect_ms = extract_json_u32(input, "detect_ms")?;
        let parse_ms = extract_json_u32(input, "parse_ms")?;
        Some(Self {
            mime_type: mime,
            content,
            metadata: Metadata::default(),
            embedded: Vec::new(),
            warnings,
            errors,
            parser_chain,
            timing: ParseTiming {
                detect_ms,
                parse_ms,
            },
        })
    }
}

fn json_str_array(items: &[String]) -> String {
    let s = items
        .iter()
        .map(|v| format!("\"{}\"", escape_json(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{s}]")
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

fn extract_json_string(input: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = input.find(&needle)? + needle.len();
    let rest = &input[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_json_nullable_string(input: &str, key: &str) -> Option<Option<String>> {
    let null_needle = format!("\"{key}\":null");
    if input.contains(&null_needle) {
        return Some(None);
    }
    extract_json_string(input, key).map(Some)
}

fn extract_json_string_array(input: &str, key: &str) -> Option<Vec<String>> {
    let needle = format!("\"{key}\":[");
    let start = input.find(&needle)? + needle.len();
    let rest = &input[start..];
    let end = rest.find(']')?;
    let body = &rest[..end];
    if body.trim().is_empty() {
        return Some(Vec::new());
    }
    Some(
        body.split(',')
            .map(|v| v.trim().trim_matches('"').to_string())
            .collect(),
    )
}

fn extract_json_u32(input: &str, key: &str) -> Option<u32> {
    let needle = format!("\"{key}\":");
    let start = input.find(&needle)? + needle.len();
    let rest = &input[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{ParseTiming, StructuredResult};

    #[test]
    fn structured_result_round_trip() {
        let src = StructuredResult {
            mime_type: "text/plain".to_string(),
            content: Some("hello".to_string()),
            warnings: vec!["w1".to_string()],
            errors: vec![],
            parser_chain: vec!["TextParser".to_string()],
            timing: ParseTiming {
                detect_ms: 1,
                parse_ms: 2,
            },
            ..StructuredResult::default()
        };
        let json = src.to_json();
        let parsed = StructuredResult::from_json(&json).expect("parse");
        assert_eq!(parsed.mime_type, "text/plain");
        assert_eq!(parsed.content.as_deref(), Some("hello"));
        assert_eq!(parsed.warnings, vec!["w1"]);
        assert_eq!(parsed.parser_chain, vec!["TextParser"]);
        assert_eq!(parsed.timing.detect_ms, 1);
        assert_eq!(parsed.timing.parse_ms, 2);
    }
}
