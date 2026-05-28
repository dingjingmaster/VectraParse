use vectraparse_core::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentHandlerOutput {
    pub content: String,
    pub metadata: Metadata,
    pub links: Vec<String>,
    pub phones: Vec<String>,
}

pub fn handle_text(input: &str, max_chars: usize) -> ContentHandlerOutput {
    let mut md = Metadata::default();
    md.insert("handler", "text");
    ContentHandlerOutput {
        content: truncate(input, max_chars),
        metadata: md,
        links: extract_links(input),
        phones: extract_phones(input),
    }
}

pub fn handle_html(input: &str, max_chars: usize) -> ContentHandlerOutput {
    let mut md = Metadata::default();
    md.insert("handler", "html");
    ContentHandlerOutput {
        content: truncate(input, max_chars),
        metadata: md,
        links: extract_links(input),
        phones: extract_phones(input),
    }
}

pub fn handle_xml(input: &str, xpath_like: Option<&str>, max_chars: usize) -> ContentHandlerOutput {
    let mut md = Metadata::default();
    md.insert("handler", "xml");
    if let Some(path) = xpath_like {
        md.insert("xpath_filter", path);
    }
    if input.contains("<xmpmeta") || input.contains("x:xmpmeta") {
        md.insert("XMP:present", "true");
    }
    ContentHandlerOutput {
        content: truncate(input, max_chars),
        metadata: md,
        links: extract_links(input),
        phones: extract_phones(input),
    }
}

fn truncate(input: &str, max_chars: usize) -> String {
    input.chars().take(max_chars).collect()
}

fn extract_links(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .filter(|s| s.starts_with("http://") || s.starts_with("https://"))
        .map(ToString::to_string)
        .collect()
}

fn extract_phones(input: &str) -> Vec<String> {
    input
        .split_whitespace()
        .filter(|s| {
            let digits = s.chars().filter(|c| c.is_ascii_digit()).count();
            digits >= 10 && (s.contains('-') || s.contains('(') || s.contains(')'))
        })
        .map(ToString::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{handle_html, handle_text, handle_xml};

    #[test]
    fn extracts_links_phones_and_truncates() {
        let out = handle_text("hello https://example.com call 138-0013-8000", 20);
        assert_eq!(out.content.len(), 20);
        assert_eq!(out.links, vec!["https://example.com"]);
        assert_eq!(out.phones, vec!["138-0013-8000"]);
    }

    #[test]
    fn xml_handler_sets_xpath_and_xmp() {
        let out = handle_xml("<xmpmeta><a/></xmpmeta>", Some("/a/b"), 200);
        assert_eq!(
            out.metadata.values("xpath_filter").and_then(|v| v.first()).map(String::as_str),
            Some("/a/b")
        );
        assert_eq!(
            out.metadata.values("XMP:present").and_then(|v| v.first()).map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn html_handler_tag() {
        let out = handle_html("<a href='https://x'>x</a>", 200);
        assert_eq!(
            out.metadata.values("handler").and_then(|v| v.first()).map(String::as_str),
            Some("html")
        );
    }
}
