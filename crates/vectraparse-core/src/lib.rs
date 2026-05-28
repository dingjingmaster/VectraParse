pub mod config;
pub mod metadata;
pub mod registry;
pub mod result;
pub mod runtime;

pub const API_VERSION: &str = "0.1.0";
pub const CAPABILITIES_JSON: &str =
    "{\"detect\":true,\"parse\":true,\"parse_file\":true,\"enhance\":false}";

fn detect_mime(input: &[u8]) -> String {
    if input.starts_with(b"%PDF-") {
        "application/pdf".to_string()
    } else if input.starts_with(b"PK\x03\x04") {
        "application/zip".to_string()
    } else if input.is_empty() {
        "application/x-empty".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

pub fn detect_json(input: &[u8]) -> String {
    let result = result::StructuredResult {
        mime_type: detect_mime(input),
        content: None,
        metadata: metadata::Metadata::default(),
        embedded: Vec::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
        parser_chain: Vec::new(),
        timing: result::ParseTiming {
            detect_ms: 0,
            parse_ms: 0,
        },
    };
    result.to_json()
}

pub fn parse_json(input: &[u8]) -> String {
    let mut md = metadata::Metadata::default();
    md.insert("Content-Length", input.len().to_string());
    let result = result::StructuredResult {
        mime_type: detect_mime(input),
        content: Some(String::new()),
        metadata: md,
        embedded: Vec::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
        parser_chain: vec!["CoreParseStub".to_string()],
        timing: result::ParseTiming {
            detect_ms: 0,
            parse_ms: 0,
        },
    };
    result.to_json()
}

pub fn detect_with_limits_json(input: &[u8], max_input_bytes: usize) -> Result<String, String> {
    let limits = runtime::ResourceLimits {
        max_input_bytes,
        ..runtime::ResourceLimits::default()
    };
    runtime::validate_input_size(input.len(), &limits).map_err(|e| format!("{e:?}"))?;
    Ok(detect_json(input))
}

pub fn parse_with_limits_json(input: &[u8], max_input_bytes: usize) -> Result<String, String> {
    let limits = runtime::ResourceLimits {
        max_input_bytes,
        ..runtime::ResourceLimits::default()
    };
    runtime::validate_input_size(input.len(), &limits).map_err(|e| format!("{e:?}"))?;
    Ok(parse_json(input))
}
