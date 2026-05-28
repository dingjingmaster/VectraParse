use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
    Io(String),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    Memory(Vec<u8>),
    File(PathBuf),
}

impl InputSource {
    pub fn from_memory(bytes: &[u8]) -> Self {
        Self::Memory(bytes.to_vec())
    }

    pub fn from_file(path: impl AsRef<Path>) -> Self {
        Self::File(path.as_ref().to_path_buf())
    }

    pub fn read_limited(&self, limits: &ResourceLimits) -> Result<Vec<u8>, ParseFailure> {
        match self {
            Self::Memory(bytes) => {
                validate_input_size(bytes.len(), limits)?;
                Ok(bytes.clone())
            }
            Self::File(path) => {
                let mut f = File::open(path).map_err(|e| ParseFailure::Io(e.to_string()))?;
                let metadata = f.metadata().map_err(|e| ParseFailure::Io(e.to_string()))?;
                let size = metadata.len() as usize;
                validate_input_size(size, limits)?;
                let mut out = Vec::with_capacity(size.min(limits.max_input_bytes));
                f.read_to_end(&mut out)
                    .map_err(|e| ParseFailure::Io(e.to_string()))?;
                validate_input_size(out.len(), limits)?;
                Ok(out)
            }
        }
    }

    pub fn read_window(&self, window: usize) -> Result<Vec<u8>, ParseFailure> {
        match self {
            Self::Memory(bytes) => Ok(bytes[..bytes.len().min(window)].to_vec()),
            Self::File(path) => {
                let mut f = File::open(path).map_err(|e| ParseFailure::Io(e.to_string()))?;
                f.seek(SeekFrom::Start(0))
                    .map_err(|e| ParseFailure::Io(e.to_string()))?;
                let mut buf = vec![0u8; window];
                let read = f
                    .read(&mut buf)
                    .map_err(|e| ParseFailure::Io(e.to_string()))?;
                buf.truncate(read);
                Ok(buf)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InputSource, ParseFailure, ResourceLimits, validate_input_size};
    use std::fs;

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

    #[test]
    fn memory_input_respects_limit() {
        let src = InputSource::from_memory(b"hello");
        let limits = ResourceLimits {
            max_input_bytes: 4,
            ..ResourceLimits::default()
        };
        let err = src.read_limited(&limits).expect_err("must fail");
        assert_eq!(err, ParseFailure::InputTooLarge { limit: 4, actual: 5 });
    }

    #[test]
    fn file_input_window_and_limit() {
        let path = "/tmp/vectraparse_runtime_test.bin";
        fs::write(path, b"abcdef").expect("write");
        let src = InputSource::from_file(path);
        assert_eq!(src.read_window(3).expect("window"), b"abc");
        let limits = ResourceLimits {
            max_input_bytes: 6,
            ..ResourceLimits::default()
        };
        assert_eq!(src.read_limited(&limits).expect("read"), b"abcdef");
        let _ = fs::remove_file(path);
    }
}
