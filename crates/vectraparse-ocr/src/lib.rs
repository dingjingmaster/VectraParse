use std::path::Path;

use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct OcrConfig {
    pub det_model_path: String,
    pub rec_model_path: String,
}

#[derive(Debug, Clone, Default)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f32,
}

pub struct TractOcrEngine {
    det: TypedRunnableModel<TypedModel>,
    rec: TypedRunnableModel<TypedModel>,
}

impl TractOcrEngine {
    pub fn load(cfg: &OcrConfig) -> TractResult<Self> {
        let det = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.det_model_path))?
            .into_optimized()?
            .into_runnable()?;
        let rec = tract_onnx::onnx()
            .model_for_path(Path::new(&cfg.rec_model_path))?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { det, rec })
    }

    pub fn infer(&self, _image_bytes: &[u8]) -> TractResult<OcrResult> {
        let _ = (&self.det, &self.rec);
        Ok(OcrResult::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_result_is_empty() {
        let out = OcrResult::default();
        assert!(out.text.is_empty());
        assert_eq!(out.confidence, 0.0);
    }
}
