#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    pub max_input_bytes: usize,
    pub timeout_ms: u32,
    pub max_embedded_depth: u16,
    pub max_embedded_files: u32,
    pub max_output_chars: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseFailure {
    InputTooLarge { limit: usize, actual: usize },
    Timeout { timeout_ms: u32 },
    Internal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseWarning {
    TruncatedOutput { max_chars: usize },
    EmbeddedLimitReached { max_files: u32 },
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 64 * 1024 * 1024,
            timeout_ms: 30_000,
            max_embedded_depth: 10,
            max_embedded_files: 1000,
            max_output_chars: 4 * 1024 * 1024,
        }
    }
}

pub fn validate_input_size(input_len: usize, limits: &ResourceLimits) -> Result<(), ParseFailure> {
    if input_len > limits.max_input_bytes {
        return Err(ParseFailure::InputTooLarge {
            limit: limits.max_input_bytes,
            actual: input_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_input_size, ParseFailure, ResourceLimits};

    #[test]
    fn reject_too_large_input() {
        let limits = ResourceLimits {
            max_input_bytes: 4,
            ..ResourceLimits::default()
        };
        let err = validate_input_size(5, &limits).expect_err("must fail");
        assert_eq!(err, ParseFailure::InputTooLarge { limit: 4, actual: 5 });
    }

    #[test]
    fn accept_within_limit() {
        let limits = ResourceLimits {
            max_input_bytes: 4,
            ..ResourceLimits::default()
        };
        validate_input_size(4, &limits).expect("must pass");
    }
}
