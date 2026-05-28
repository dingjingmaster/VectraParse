use crate::embedded::{EmbeddedLimits, EmbeddedNode, collect_embedded};

pub trait ContainerExtractor {
    fn extract(&self, input: &[u8], limits: &EmbeddedLimits) -> (Vec<EmbeddedNode>, Vec<String>);
}

pub struct ParserContainerExtractor;
impl ContainerExtractor for ParserContainerExtractor {
    fn extract(&self, input: &[u8], limits: &EmbeddedLimits) -> (Vec<EmbeddedNode>, Vec<String>) {
        collect_embedded(input, limits)
    }
}

pub trait Embedder {
    fn enabled(&self) -> bool;
    fn process(&self, nodes: Vec<EmbeddedNode>) -> Vec<EmbeddedNode>;
}

pub struct ExternalEmbedder {
    pub enabled: bool,
}

impl Embedder for ExternalEmbedder {
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn process(&self, nodes: Vec<EmbeddedNode>) -> Vec<EmbeddedNode> {
        if !self.enabled {
            return Vec::new();
        }
        nodes
    }
}

pub fn extract_with_embedder(
    extractor: &dyn ContainerExtractor,
    embedder: &dyn Embedder,
    input: &[u8],
    limits: &EmbeddedLimits,
) -> (Vec<EmbeddedNode>, Vec<String>) {
    let (nodes, mut warnings) = extractor.extract(input, limits);
    if !embedder.enabled() {
        warnings.push("embedder-disabled".to_string());
        return (Vec::new(), warnings);
    }
    (embedder.process(nodes), warnings)
}

#[cfg(test)]
mod tests {
    use super::{ExternalEmbedder, ParserContainerExtractor, extract_with_embedder};
    use crate::embedded::EmbeddedLimits;

    #[test]
    fn extractor_and_embedder_enabled() {
        let extractor = ParserContainerExtractor;
        let embedder = ExternalEmbedder { enabled: true };
        let input = b"[[EMBED:a]]x[[/EMBED]]";
        let (nodes, warnings) =
            extract_with_embedder(&extractor, &embedder, input, &EmbeddedLimits::default());
        assert_eq!(warnings.len(), 0);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn extractor_embedder_disabled() {
        let extractor = ParserContainerExtractor;
        let embedder = ExternalEmbedder { enabled: false };
        let input = b"[[EMBED:a]]x[[/EMBED]]";
        let (nodes, warnings) =
            extract_with_embedder(&extractor, &embedder, input, &EmbeddedLimits::default());
        assert_eq!(nodes.len(), 0);
        assert!(warnings.iter().any(|w| w == "embedder-disabled"));
    }
}
