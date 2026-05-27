pub mod config;
pub mod metadata;
pub mod runtime;

pub const API_VERSION: &str = "0.1.0";
pub const CAPABILITIES_JSON: &str =
    "{\"detect\":true,\"parse\":true,\"parse_file\":false,\"enhance\":false}";

fn detect_mime(input: &[u8]) -> &'static str {
    if input.starts_with(b"%PDF-") {
        "application/pdf"
    } else if input.starts_with(b"PK\x03\x04") {
        "application/zip"
    } else {
        "application/octet-stream"
    }
}

pub fn detect_json(input: &[u8]) -> String {
    let mime = detect_mime(input);
    format!(
        "{{\"mime_type\":\"{mime}\",\"metadata\":{{}},\"content\":null,\"embedded\":[],\"warnings\":[],\"error\":null}}"
    )
}

pub fn parse_json(input: &[u8]) -> String {
    let mime = detect_mime(input);
    format!(
        "{{\"mime_type\":\"{mime}\",\"metadata\":{{\"Content-Length\":[\"{}\"]}},\"content\":\"\",\"embedded\":[],\"warnings\":[],\"error\":null}}",
        input.len()
    )
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
