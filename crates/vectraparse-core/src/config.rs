#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseConfig {
    pub enabled_parsers: Vec<String>,
    pub max_bytes: usize,
    pub timeout_ms: u32,
    pub external_command: Option<String>,
    pub model_path: Option<String>,
    pub service_key_ref: Option<String>,
    pub include_metadata: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    InvalidLine(String),
    UnknownKey(String),
    InvalidNumber(String),
    InvalidBoolean(String),
}

impl ParseConfig {
    pub fn from_kv(input: &str) -> Result<Self, ConfigError> {
        let mut cfg = Self::default();
        for raw in input.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| ConfigError::InvalidLine(line.to_string()))?;
            let key = key.trim();
            let value = value.trim();
            match key {
                "enabled_parsers" => {
                    cfg.enabled_parsers = value
                        .split(',')
                        .map(str::trim)
                        .filter(|v| !v.is_empty())
                        .map(ToString::to_string)
                        .collect();
                }
                "max_bytes" => {
                    cfg.max_bytes = value
                        .parse()
                        .map_err(|_| ConfigError::InvalidNumber(value.to_string()))?;
                }
                "timeout_ms" => {
                    cfg.timeout_ms = value
                        .parse()
                        .map_err(|_| ConfigError::InvalidNumber(value.to_string()))?;
                }
                "external_command" => {
                    cfg.external_command = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "model_path" => {
                    cfg.model_path = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "service_key_ref" => {
                    cfg.service_key_ref = if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    };
                }
                "include_metadata" => {
                    cfg.include_metadata = parse_bool(value)?;
                }
                _ => return Err(ConfigError::UnknownKey(key.to_string())),
            }
        }
        Ok(cfg)
    }
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            enabled_parsers: vec![
                "text".to_string(),
                "html".to_string(),
                "xml".to_string(),
                "pdf".to_string(),
            ],
            max_bytes: 64 * 1024 * 1024,
            timeout_ms: 30_000,
            external_command: None,
            model_path: None,
            service_key_ref: None,
            include_metadata: true,
        }
    }
}

fn parse_bool(value: &str) -> Result<bool, ConfigError> {
    match value {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(ConfigError::InvalidBoolean(value.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigError, ParseConfig};

    #[test]
    fn parse_valid_config() {
        let input = r#"
enabled_parsers=text,html,pdf
max_bytes=4096
timeout_ms=250
external_command=tesseract
model_path=/opt/models
service_key_ref=env:API_KEY
include_metadata=false
"#;
        let cfg = ParseConfig::from_kv(input).expect("parse config");
        assert_eq!(cfg.enabled_parsers, vec!["text", "html", "pdf"]);
        assert_eq!(cfg.max_bytes, 4096);
        assert_eq!(cfg.timeout_ms, 250);
        assert_eq!(cfg.external_command.as_deref(), Some("tesseract"));
        assert_eq!(cfg.model_path.as_deref(), Some("/opt/models"));
        assert_eq!(cfg.service_key_ref.as_deref(), Some("env:API_KEY"));
        assert!(!cfg.include_metadata);
    }

    #[test]
    fn reject_invalid_number() {
        let input = "max_bytes=oops";
        let err = ParseConfig::from_kv(input).expect_err("invalid");
        assert_eq!(err, ConfigError::InvalidNumber("oops".to_string()));
    }

    #[test]
    fn reject_unknown_key() {
        let input = "unexpected_key=1";
        let err = ParseConfig::from_kv(input).expect_err("invalid");
        assert_eq!(err, ConfigError::UnknownKey("unexpected_key".to_string()));
    }
}
