pub mod content_handler;
pub mod embedded;
pub mod extractor;
pub mod security;

use vectraparse_core::metadata::Metadata;
use vectraparse_mime::detect_encoding;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParseOutcome {
    pub content: Option<String>,
    pub metadata: Metadata,
    pub warnings: Vec<String>,
    pub parser_chain: Vec<String>,
}

pub trait Parser: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, media_type: &str) -> bool;
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome>;
}

pub struct CompositeParser {
    parsers: Vec<Box<dyn Parser>>,
}

impl CompositeParser {
    pub fn new(parsers: Vec<Box<dyn Parser>>) -> Self {
        Self { parsers }
    }

    pub fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        let mut supplement = ParseOutcome::default();
        for parser in &self.parsers {
            if !parser.supports(media_type) {
                continue;
            }
            if let Some(mut out) = parser.parse(input, media_type) {
                out.parser_chain.insert(0, parser.name().to_string());
                if out.content.is_some() {
                    let mut warnings = std::mem::take(&mut out.warnings);
                    supplement.warnings.append(&mut warnings);
                    for k in out.metadata.keys().map(ToString::to_string).collect::<Vec<_>>() {
                        if let Some(vals) = out.metadata.values(&k) {
                            for v in vals {
                                supplement.metadata.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    if out.content.is_some() {
                        out.warnings = supplement.warnings.clone();
                        for k in supplement
                            .metadata
                            .keys()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                        {
                            if let Some(vals) = supplement.metadata.values(&k) {
                                for v in vals {
                                    out.metadata.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                    return Some(out);
                }
                supplement.warnings.extend(out.warnings);
                for k in out.metadata.keys().map(ToString::to_string).collect::<Vec<_>>() {
                    if let Some(vals) = out.metadata.values(&k) {
                        for v in vals {
                            supplement.metadata.insert(k.clone(), v.clone());
                        }
                    }
                }
                supplement.parser_chain.extend(out.parser_chain);
            }
        }
        None
    }

    pub fn parse_multiple(&self, input: &[u8], media_type: &str) -> Vec<ParseOutcome> {
        self.parsers
            .iter()
            .filter(|p| p.supports(media_type))
            .filter_map(|p| {
                p.parse(input, media_type).map(|mut out| {
                    out.parser_chain.insert(0, p.name().to_string());
                    out
                })
            })
            .collect()
    }
}

pub struct TxtParser;
impl Parser for TxtParser {
    fn name(&self) -> &'static str {
        "TxtParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/plain"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return Some(ParseOutcome {
                content: Some(String::new()),
                metadata: Metadata::default(),
                warnings: vec!["empty-input".to_string()],
                parser_chain: Vec::new(),
            });
        }
        let enc = detect_encoding(input);
        if enc == "binary" {
            return None;
        }
        let content = match enc {
            "utf-8" => String::from_utf8(input.to_vec()).ok()?,
            "utf-16le" => decode_utf16le(input)?,
            "utf-16be" => decode_utf16be(input)?,
            _ => String::from_utf8_lossy(input).to_string(),
        };
        let mut metadata = Metadata::default();
        metadata.insert("parser", "TxtParser");
        metadata.insert("encoding", enc);
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct TextAndCsvParser;
impl Parser for TextAndCsvParser {
    fn name(&self) -> &'static str {
        "TextAndCsvParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/csv" || media_type == "text/tab-separated-values"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let delimiter = detect_delimiter(&content);
        let (bad_rows, row_count) = analyze_delimited_rows(&content, delimiter);
        let mut metadata = Metadata::default();
        metadata.insert("parser", "TextAndCsvParser");
        metadata.insert("csv.delimiter", delimiter.to_string());
        metadata.insert("csv.rows", row_count.to_string());
        metadata.insert("csv.bad_rows", bad_rows.to_string());
        if content.contains('"') {
            metadata.insert("csv.quote_style", "double-quote");
        }
        let mut warnings = Vec::new();
        if bad_rows > 0 {
            warnings.push("csv-bad-row-detected".to_string());
        }
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct HtmlParser;
impl Parser for HtmlParser {
    fn name(&self) -> &'static str {
        "HtmlParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/html"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let title = extract_between(&content, "<title>", "</title>");
        let charset = extract_html_charset(&lower);
        let links = extract_html_links(&content);
        let body = strip_html_tags(&content);
        let mut metadata = Metadata::default();
        metadata.insert("parser", "HtmlParser");
        if let Some(t) = title {
            metadata.insert("html.title", t);
        }
        if let Some(cs) = charset {
            metadata.insert("html.charset", cs);
        }
        for (k, v) in extract_meta_pairs(&content) {
            metadata.insert(format!("html.meta.{k}"), v);
        }
        for link in &links {
            metadata.insert("html.link", link.clone());
        }
        let mut warnings = Vec::new();
        if content.len() > 512 * 1024 || content.matches('<').count() > 10_000 {
            warnings.push("html-depth-limit-applied".to_string());
        }
        Some(ParseOutcome {
            content: Some(body),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct MetadataOnlyParser;
impl Parser for MetadataOnlyParser {
    fn name(&self) -> &'static str {
        "MetadataOnlyParser"
    }
    fn supports(&self, _media_type: &str) -> bool {
        true
    }
    fn parse(&self, _input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let mut metadata = Metadata::default();
        metadata.insert("supplement", "true");
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings: vec!["supplement-parser".to_string()],
            parser_chain: vec!["MetadataOnlyParser".to_string()],
        })
    }
}

pub struct XmlParser;
impl Parser for XmlParser {
    fn name(&self) -> &'static str {
        "XmlParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/xml" || media_type.ends_with("+xml")
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        if lower.contains("<!doctype") && lower.contains("<!entity") {
            return Some(ParseOutcome {
                content: None,
                metadata: Metadata::default(),
                warnings: vec!["xxe-blocked".to_string()],
                parser_chain: Vec::new(),
            });
        }
        let root = extract_xml_root(&content)?;
        let mut metadata = Metadata::default();
        metadata.insert("parser", "XmlParser");
        metadata.insert("xml.root", root.clone());
        if root.eq_ignore_ascii_case("dc") {
            metadata.insert("xml.profile", "DcXML");
        }
        if root.eq_ignore_ascii_case("fictionbook") {
            metadata.insert("xml.profile", "FictionBook");
        }
        // placeholder XPath behavior: keep text after matching tag if present
        let text = strip_html_tags(&content);
        Some(ParseOutcome {
            content: Some(text),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct SourceCodeParser;
impl Parser for SourceCodeParser {
    fn name(&self) -> &'static str {
        "SourceCodeParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "text/x-source"
            || media_type == "text/x-rust"
            || media_type == "text/x-python"
            || media_type == "text/x-java"
            || media_type == "application/javascript"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lang = detect_source_language(&content);
        let mut metadata = Metadata::default();
        metadata.insert("parser", "SourceCodeParser");
        metadata.insert("source.language", lang);
        metadata.insert("source.lines", content.lines().count().to_string());
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct StringsParser;
impl Parser for StringsParser {
    fn name(&self) -> &'static str {
        "StringsParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/octet-stream"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let mut strings = extract_ascii_strings(input, 4, 4096);
        strings.extend(extract_latin1_strings(input, 4, 4096));
        strings.sort();
        strings.dedup();
        if strings.is_empty() {
            return None;
        }
        let joined = strings.join("\n");
        let truncated: String = joined.chars().take(4096).collect();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "StringsParser");
        metadata.insert("strings.count", strings.len().to_string());
        metadata.insert("strings.charset", "ascii+latin1");
        Some(ParseOutcome {
            content: Some(truncated),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct FeedParser;
impl Parser for FeedParser {
    fn name(&self) -> &'static str {
        "FeedParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/rss+xml"
            || media_type == "application/atom+xml"
            || media_type == "application/xml"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "FeedParser");
        let mut warnings = Vec::new();
        let feed_type = if lower.contains("<rss") {
            "rss"
        } else if lower.contains("<feed") {
            "atom"
        } else {
            "unknown"
        };
        metadata.insert("feed.type", feed_type);
        if feed_type == "unknown" {
            warnings.push("feed-fallback-plain-xml".to_string());
            return Some(ParseOutcome {
                content: Some(strip_html_tags(&content)),
                metadata,
                warnings,
                parser_chain: Vec::new(),
            });
        }
        if !lower.contains("</rss>") && !lower.contains("</feed>") {
            warnings.push("feed-malformed-xml".to_string());
            return Some(ParseOutcome {
                content: Some(strip_html_tags(&content)),
                metadata,
                warnings,
                parser_chain: Vec::new(),
            });
        }
        let title = extract_between(&content, "<title>", "</title>").unwrap_or_default();
        if !title.is_empty() {
            metadata.insert("feed.title", title);
        }
        let links = extract_feed_links(&content);
        for l in &links {
            metadata.insert("feed.link", l.clone());
        }
        metadata.insert("feed.link_count", links.len().to_string());
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

fn decode_utf16le(input: &[u8]) -> Option<String> {
    if !input.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for chunk in input.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    String::from_utf16(&out).ok()
}

fn decode_utf16be(input: &[u8]) -> Option<String> {
    if !input.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for chunk in input.chunks_exact(2) {
        out.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }
    String::from_utf16(&out).ok()
}

fn detect_delimiter(content: &str) -> char {
    let sample = content.lines().take(8).collect::<Vec<_>>().join("\n");
    let comma = sample.matches(',').count();
    let tab = sample.matches('\t').count();
    let semi = sample.matches(';').count();
    if tab >= comma && tab >= semi {
        '\t'
    } else if semi > comma {
        ';'
    } else {
        ','
    }
}

fn analyze_delimited_rows(content: &str, delimiter: char) -> (usize, usize) {
    let mut expected_cols: Option<usize> = None;
    let mut bad_rows = 0usize;
    let mut row_count = 0usize;
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        row_count += 1;
        let cols = split_csv_like(line, delimiter).len();
        match expected_cols {
            None => expected_cols = Some(cols),
            Some(exp) if exp != cols => bad_rows += 1,
            _ => {}
        }
    }
    (bad_rows, row_count)
}

fn split_csv_like(line: &str, delimiter: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_quote = false;
    for ch in line.chars() {
        if ch == '"' {
            in_quote = !in_quote;
            continue;
        }
        if ch == delimiter && !in_quote {
            out.push(buf.clone());
            buf.clear();
            continue;
        }
        buf.push(ch);
    }
    out.push(buf);
    out
}

fn extract_between(s: &str, start: &str, end: &str) -> Option<String> {
    let i = s.to_ascii_lowercase().find(&start.to_ascii_lowercase())?;
    let j0 = i + start.len();
    let rest = &s[j0..];
    let j = rest.to_ascii_lowercase().find(&end.to_ascii_lowercase())?;
    Some(rest[..j].trim().to_string())
}

fn extract_html_charset(lower: &str) -> Option<String> {
    let idx = lower.find("charset=")?;
    let mut rest = &lower[idx + "charset=".len()..];
    rest = rest.trim_start();
    if let Some(stripped) = rest.strip_prefix('"').or_else(|| rest.strip_prefix('\'')) {
        rest = stripped;
    }
    let end = rest
        .find(|c: char| c == '"' || c == '\'' || c == '>' || c.is_whitespace() || c == ';')
        .unwrap_or(rest.len());
    let v = rest[..end].trim();
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

fn extract_html_links(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(pos) = s[i..].find("href=") {
        let start = i + pos + "href=".len();
        let bytes = s.as_bytes();
        if start >= s.len() {
            break;
        }
        let quote = bytes[start] as char;
        if quote != '"' && quote != '\'' {
            i = start;
            continue;
        }
        let rest = &s[start + 1..];
        if let Some(end) = rest.find(quote) {
            out.push(rest[..end].to_string());
            i = start + 1 + end + 1;
        } else {
            break;
        }
    }
    out
}

fn extract_meta_pairs(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for part in s.split("<meta").skip(1) {
        let tag = match part.find('>') {
            Some(i) => &part[..i],
            None => continue,
        };
        let name = extract_attr(tag, "name");
        let content = extract_attr(tag, "content");
        if let (Some(n), Some(c)) = (name, content) {
            out.push((n, c));
        }
    }
    out
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{attr}=");
    let i = lower.find(&needle)?;
    let start = i + needle.len();
    let b = tag.as_bytes();
    if start >= tag.len() {
        return None;
    }
    let quote = b[start] as char;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let rest = &tag[start + 1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_xml_root(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if i + 1 < bytes.len() && (bytes[i + 1] == b'?' || bytes[i + 1] == b'!') {
                i += 1;
            } else {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() {
                    let c = bytes[j] as char;
                    if c.is_whitespace() || c == '>' || c == '/' {
                        break;
                    }
                    j += 1;
                }
                if j > start {
                    return Some(s[start..j].to_string());
                }
            }
        }
        i += 1;
    }
    None
}

fn detect_source_language(content: &str) -> &'static str {
    if content.contains("fn main(") || content.contains("use std::") {
        "rust"
    } else if content.contains("def ") || content.contains("import ") {
        "python"
    } else if content.contains("public class ") || content.contains("package ") {
        "java"
    } else if content.contains("function ") || content.contains("const ") {
        "javascript"
    } else {
        "plain"
    }
}

fn extract_feed_links(content: &str) -> Vec<String> {
    let mut out = extract_html_links(content);
    let mut i = 0usize;
    while let Some(pos) = content[i..].find("<link") {
        let start = i + pos;
        let rest = &content[start..];
        let end = match rest.find('>') {
            Some(v) => v,
            None => break,
        };
        let tag = &rest[..end];
        if let Some(href) = extract_attr(tag, "href") {
            out.push(href);
        }
        i = start + end + 1;
    }
    out.sort();
    out.dedup();
    out
}

fn extract_ascii_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || b == b'\t' || b == b' ';
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            out.push(String::from_utf8_lossy(&buf).to_string());
        }
        buf.clear();
        if out.iter().map(|s| s.len()).sum::<usize>() >= max_chars {
            break;
        }
    }
    if buf.len() >= min_len && out.iter().map(|s| s.len()).sum::<usize>() < max_chars {
        out.push(String::from_utf8_lossy(&buf).to_string());
    }
    out
}

fn extract_latin1_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || (0xa0..=0xff).contains(&b);
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            out.push(buf.iter().map(|c| *c as char).collect::<String>());
        }
        buf.clear();
        if out.iter().map(|s| s.len()).sum::<usize>() >= max_chars {
            break;
        }
    }
    if buf.len() >= min_len && out.iter().map(|s| s.len()).sum::<usize>() < max_chars {
        out.push(buf.iter().map(|c| *c as char).collect::<String>());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        CompositeParser, HtmlParser, MetadataOnlyParser, Parser, TextAndCsvParser, TxtParser,
        FeedParser, SourceCodeParser, StringsParser, XmlParser,
    };

    #[test]
    fn mime_to_parser_mapping_and_fallback() {
        let composite = CompositeParser::new(vec![
            Box::new(MetadataOnlyParser),
            Box::new(HtmlParser),
            Box::new(TxtParser),
            Box::new(TextAndCsvParser),
            Box::new(XmlParser),
        ]);
        let text = composite
            .parse(b"hello", "text/plain")
            .expect("text parser should parse");
        assert_eq!(text.content.as_deref(), Some("hello"));
        assert!(text.parser_chain.contains(&"TxtParser".to_string()));
        assert!(text.metadata.values("supplement").is_some());
        assert!(composite.parse(b"\xFF\xFE", "application/pdf").is_none());
    }

    #[test]
    fn multiple_parser_dispatch_returns_all_matches() {
        let composite = CompositeParser::new(vec![
            Box::new(TxtParser),
            Box::new(TextAndCsvParser),
            Box::new(MetadataOnlyParser),
        ]);
        let all = composite.parse_multiple(b"a,b,c", "text/csv");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn txt_parser_handles_empty_and_binary() {
        let p = TxtParser;
        let empty = p.parse(b"", "text/plain").expect("empty");
        assert_eq!(empty.content.as_deref(), Some(""));
        assert!(empty.warnings.iter().any(|w| w == "empty-input"));
        assert!(p.parse(&[0, 159, 146, 150], "text/plain").is_none());
    }

    #[test]
    fn csv_tsv_parser_handles_dialect_escape_and_bad_rows() {
        let p = TextAndCsvParser;
        let out = p
            .parse(b"col1,col2\n\"a,b\",c\nx", "text/csv")
            .expect("csv");
        assert_eq!(
            out.metadata
                .values("csv.delimiter")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some(",")
        );
        assert_eq!(
            out.metadata
                .values("csv.bad_rows")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("1")
        );
        assert!(out.warnings.iter().any(|w| w == "csv-bad-row-detected"));

        let tsv = p.parse(b"a\tb\n1\t2", "text/tab-separated-values").expect("tsv");
        assert_eq!(
            tsv.metadata
                .values("csv.delimiter")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("\t")
        );
    }

    #[test]
    fn html_parser_extracts_title_meta_links_charset_and_body() {
        let p = HtmlParser;
        let out = p
            .parse(
                br#"<html><head><title>Hello</title><meta charset="utf-8"><meta name="author" content="alice"></head><body><a href="https://example.com">x</a>text</body></html>"#,
                "text/html",
            )
            .expect("html");
        assert_eq!(
            out.metadata
                .values("html.title")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("Hello")
        );
        assert_eq!(
            out.metadata
                .values("html.charset")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("utf-8")
        );
        assert_eq!(
            out.metadata
                .values("html.meta.author")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("alice")
        );
        assert_eq!(
            out.metadata
                .values("html.link")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("https://example.com")
        );
        assert!(out.content.as_deref().unwrap_or("").contains("text"));
    }

    #[test]
    fn xml_parser_extracts_root_and_blocks_xxe() {
        let p = XmlParser;
        let out = p
            .parse(b"<?xml version='1.0'?><FictionBook><body>x</body></FictionBook>", "application/xml")
            .expect("xml");
        assert_eq!(
            out.metadata
                .values("xml.root")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("FictionBook")
        );
        assert_eq!(
            out.metadata
                .values("xml.profile")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("FictionBook")
        );

        let blocked = p
            .parse(
                b"<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>",
                "application/xml",
            )
            .expect("xxe result");
        assert!(blocked.warnings.iter().any(|w| w == "xxe-blocked"));
        assert!(blocked.content.is_none());
    }

    #[test]
    fn source_code_parser_detects_language() {
        let p = SourceCodeParser;
        let out = p
            .parse(b"fn main() { println!(\"hi\"); }", "text/x-source")
            .expect("source");
        assert_eq!(
            out.metadata
                .values("source.language")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("rust")
        );
    }

    #[test]
    fn strings_parser_extracts_ascii_runs() {
        let p = StringsParser;
        let out = p
            .parse(b"\x00ABCDEF\x00xyz\x00", "application/octet-stream")
            .expect("strings");
        assert!(out.content.as_deref().unwrap_or("").contains("ABCDEF"));
        assert_eq!(
            out.metadata
                .values("strings.charset")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("ascii+latin1")
        );
    }

    #[test]
    fn feed_parser_supports_rss_atom_and_malformed_fallback() {
        let p = FeedParser;
        let rss = p
            .parse(
                br#"<?xml version="1.0"?><rss><channel><title>T</title><link>https://a</link></channel></rss>"#,
                "application/rss+xml",
            )
            .expect("rss");
        assert_eq!(
            rss.metadata
                .values("feed.type")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("rss")
        );
        let atom = p
            .parse(
                br#"<?xml version="1.0"?><feed><title>A</title><link href="https://b"/></feed>"#,
                "application/atom+xml",
            )
            .expect("atom");
        assert_eq!(
            atom.metadata
                .values("feed.type")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("atom")
        );
        let bad = p
            .parse(br#"<?xml version="1.0"?><feed><title>bad"#, "application/xml")
            .expect("bad");
        assert!(bad.warnings.iter().any(|w| w == "feed-malformed-xml"));
    }
}
