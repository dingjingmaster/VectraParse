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

pub struct DerivedTextParser;
impl Parser for DerivedTextParser {
    fn name(&self) -> &'static str {
        "DerivedTextParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/x-xliff+xml"
            || media_type == "text/x-dif"
            || media_type == "text/x-envi-header"
            || media_type == "application/x-iptc-anpa"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "DerivedTextParser");
        let format = if lower.contains("<xliff") {
            "xliff"
        } else if lower.contains("table") && lower.contains("vectors") {
            "dif"
        } else if lower.contains("envi") && lower.contains("samples") {
            "envi-header"
        } else if lower.contains("anpa") || lower.contains("iptc") {
            "iptc-anpa"
        } else {
            "derived-text-unknown"
        };
        metadata.insert("derived.format", format);
        Some(ParseOutcome {
            content: Some(content),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct LightweightSpecializedParser;
impl Parser for LightweightSpecializedParser {
    fn name(&self) -> &'static str {
        "LightweightSpecializedParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/applefile"
            || media_type == "application/x-plist"
            || media_type == "application/x-bplist"
            || media_type == "application/xml"
            || media_type.ends_with("+xml")
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "LightweightSpecializedParser");
        let profile = if lower.contains("applesingle") || lower.contains("appledouble") {
            "AppleSingle"
        } else if lower.contains("<plist") || lower.contains("bplist00") {
            "PList"
        } else if lower.contains("<fictionbook") {
            "FictionBook"
        } else if lower.contains("<dc") || lower.contains("dublin core") {
            "DcXML"
        } else {
            "unknown-lightweight"
        };
        metadata.insert("lightweight.profile", profile);
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct PackageParser;
impl Parser for PackageParser {
    fn name(&self) -> &'static str {
        "PackageParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/zip"
                | "application/x-tar"
                | "application/gzip"
                | "application/x-bzip2"
                | "application/x-xz"
                | "application/zstd"
                | "application/x-7z-compressed"
                | "application/vnd.rar"
                | "application/x-rar-compressed"
        )
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let pkg = detect_package_kind(input)?;
        let mut metadata = Metadata::default();
        metadata.insert("parser", "PackageParser");
        metadata.insert("package.kind", pkg.to_string());
        metadata.insert("package.input_bytes", input.len().to_string());
        let (entry_count, inflated_bytes) = estimate_archive_stats(input);
        metadata.insert("package.entry_count", entry_count.to_string());
        metadata.insert("package.estimated_inflated_bytes", inflated_bytes.to_string());
        let mut warnings = Vec::new();
        if inflated_bytes > input.len().saturating_mul(200) {
            warnings.push("package-expansion-ratio-limit".to_string());
        }
        if entry_count > 1000 {
            warnings.push("package-entry-limit".to_string());
        }
        if input.windows(6).filter(|w| *w == b"[[DIR:").count() > 16 {
            warnings.push("package-depth-limit".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct OoxmlParser;
impl Parser for OoxmlParser {
    fn name(&self) -> &'static str {
        "OoxmlParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/x-tika-ooxml"
            || media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            || media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            || media_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            || media_type == "application/xml"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "OoxmlParser");
        let doc_kind = if lower.contains("word/") || lower.contains("word/document.xml") {
            "docx"
        } else if lower.contains("xl/") || lower.contains("workbook.xml") {
            "xlsx"
        } else if lower.contains("ppt/") || lower.contains("presentation.xml") {
            "pptx"
        } else if lower.contains("wordml") {
            "wordml"
        } else if lower.contains("spreadsheetml") {
            "spreadsheetml"
        } else {
            "ooxml-unknown"
        };
        metadata.insert("ooxml.kind", doc_kind);
        if lower.contains("_rels/.rels") || lower.contains(".rels") {
            metadata.insert("ooxml.relationships", "true");
        }
        if lower.contains("docprops/core.xml") {
            metadata.insert("ooxml.core_props", "true");
        }
        let embedded_count = lower.matches("embeddings/").count() + lower.matches("oleobject").count();
        metadata.insert("ooxml.embedded_count", embedded_count.to_string());
        let mut warnings = Vec::new();
        if embedded_count > 64 {
            warnings.push("ooxml-embedded-limit".to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct OdfParser;
impl Parser for OdfParser {
    fn name(&self) -> &'static str {
        "OdfParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/vnd.oasis.opendocument"
            || media_type == "application/vnd.oasis.opendocument.text"
            || media_type == "application/vnd.oasis.opendocument.spreadsheet"
            || media_type == "application/vnd.oasis.opendocument.presentation"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "OdfParser");
        let kind = if lower.contains("mimetypeapplication/vnd.oasis.opendocument.text") {
            "odt"
        } else if lower.contains("mimetypeapplication/vnd.oasis.opendocument.spreadsheet") {
            "ods"
        } else if lower.contains("mimetypeapplication/vnd.oasis.opendocument.presentation") {
            "odp"
        } else {
            "odf"
        };
        metadata.insert("odf.kind", kind);
        if lower.contains("meta.xml") {
            metadata.insert("odf.meta", "true");
        }
        if lower.contains("manifest.xml") {
            metadata.insert("odf.manifest", "true");
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct EpubParser;
impl Parser for EpubParser {
    fn name(&self) -> &'static str {
        "EpubParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/epub+zip"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "EpubParser");
        if lower.contains("meta-inf/container.xml") {
            metadata.insert("epub.container", "true");
        }
        if lower.contains("spine") || lower.contains("<itemref") {
            metadata.insert("epub.spine", "true");
        }
        if lower.contains("dc:title") || lower.contains("dc:creator") {
            metadata.insert("epub.metadata", "true");
        }
        let embedded = lower.matches("images/").count()
            + lower.matches("audio/").count()
            + lower.matches("video/").count();
        metadata.insert("epub.embedded_resources", embedded.to_string());
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct IworkParser;
impl Parser for IworkParser {
    fn name(&self) -> &'static str {
        "IworkParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/x-iwork-package"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "IworkParser");
        let kind = if lower.contains("index/document.iwa") || lower.contains("pages") {
            "pages"
        } else if lower.contains("index/tables/") || lower.contains("numbers") {
            "numbers"
        } else if lower.contains("keynote") || lower.contains("index/metadata.iwa") {
            "keynote"
        } else {
            "iwork-unknown"
        };
        metadata.insert("iwork.kind", kind);
        let iwa_count = lower.matches(".iwa").count();
        metadata.insert("iwork.iwa_count", iwa_count.to_string());
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct PdfParser;
impl Parser for PdfParser {
    fn name(&self) -> &'static str {
        "PdfParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/pdf"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if !input.starts_with(b"%PDF-") {
            return Some(ParseOutcome {
                content: None,
                metadata: Metadata::default(),
                warnings: vec!["pdf-invalid-header".to_string()],
                parser_chain: Vec::new(),
            });
        }
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "PdfParser");
        if let Some(v) = extract_between(&content, "%PDF-", "\n") {
            metadata.insert("pdf.version", v);
        }
        let attachment_count = lower.matches("/embeddedfile").count();
        metadata.insert("pdf.attachment_count", attachment_count.to_string());
        if lower.contains("/encrypt") {
            metadata.insert("pdf.encrypted", "true");
        } else {
            metadata.insert("pdf.encrypted", "false");
        }
        if lower.contains("/p ") || lower.contains("accesspermissions") {
            metadata.insert("pdf.permissions", "present");
        }
        let mut warnings = Vec::new();
        if lower.contains("ocr:image") || lower.contains("scan-only") {
            warnings.push("pdf-ocr-hook-suggested".to_string());
        }
        if lower.contains("preflight:error") || lower.contains("xref corruption") {
            warnings.push("pdf-preflight-warning".to_string());
        }
        if input.len() > 16 * 1024 * 1024 {
            warnings.push("pdf-large-file".to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct OleLegacyParser;
impl Parser for OleLegacyParser {
    fn name(&self) -> &'static str {
        "OleLegacyParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/x-tika-msoffice"
            || media_type == "application/msword"
            || media_type == "application/vnd.ms-excel"
            || media_type == "application/vnd.ms-powerpoint"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if !input.starts_with(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1") {
            return None;
        }
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "OleLegacyParser");
        let kind = if lower.contains("worddocument") {
            "doc"
        } else if lower.contains("workbook") || lower.contains("book") {
            "xls"
        } else if lower.contains("powerpoint document") {
            "ppt"
        } else if lower.contains("biff") || lower.contains("oldexcel") {
            "oldexcel"
        } else if lower.contains("ownerfile") || lower.contains("~$") {
            "msoffice-ownerfile"
        } else {
            "ole-unknown"
        };
        metadata.insert("ole.kind", kind);
        let mut warnings = Vec::new();
        if lower.contains("vba") || lower.contains("macros") {
            warnings.push("ole-macro-present".to_string());
        }
        if lower.contains("ole10native") || lower.contains("embedded object") {
            warnings.push("ole-embedded-object".to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct MsSpecialParser;
impl Parser for MsSpecialParser {
    fn name(&self) -> &'static str {
        "MsSpecialParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/x-tika-msoffice"
            || media_type == "application/vnd.ms-outlook"
            || media_type == "application/x-tnef"
            || media_type == "image/emf"
            || media_type == "image/wmf"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "MsSpecialParser");
        let kind = if lower.contains("onenote") || lower.contains(".one") {
            "onenote"
        } else if lower.contains("standard jet db") || lower.contains("msysobjects") {
            "access"
        } else if lower.contains("tnef") || lower.contains("winmail.dat") {
            "tnef"
        } else if lower.contains(" emf") || lower.contains("emf+") {
            "emf"
        } else if lower.contains("wmf") || lower.contains("metafile") {
            "wmf"
        } else if lower.contains("__substg1.0_") {
            "msg"
        } else if lower.contains("!bdn") || lower.contains("pst") {
            "pst"
        } else {
            "ms-special-unknown"
        };
        metadata.insert("ms.kind", kind);
        if kind == "msg" || kind == "pst" {
            let attachments = lower.matches("attach").count();
            metadata.insert("ms.mail.attachments", attachments.to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct RtfParser;
impl Parser for RtfParser {
    fn name(&self) -> &'static str {
        "RtfParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/rtf" || media_type == "text/rtf"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8(input.to_vec()).ok()?;
        if !content.starts_with("{\\rtf") {
            return None;
        }
        let mut metadata = Metadata::default();
        metadata.insert("parser", "RtfParser");
        let object_count = content.matches("\\object").count() + content.matches("\\objdata").count();
        metadata.insert("rtf.object_count", object_count.to_string());
        let embedded_levels = content.matches("\\objdata").count();
        metadata.insert("rtf.object_depth_hint", embedded_levels.to_string());
        let plain = strip_rtf_control_words(&content);
        let mut warnings = Vec::new();
        if object_count > 32 {
            warnings.push("rtf-object-limit".to_string());
        }
        Some(ParseOutcome {
            content: Some(plain),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct LegacyDocParser;
impl Parser for LegacyDocParser {
    fn name(&self) -> &'static str {
        "LegacyDocParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/x-hwp"
                | "application/vnd.ms-htmlhelp"
                | "application/wordperfect"
                | "application/x-quattro-pro"
                | "application/octet-stream"
        )
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "LegacyDocParser");
        let kind = if lower.contains("hwp document") || lower.contains("hangul word processor") {
            "hwp"
        } else if lower.contains("itsf") || lower.contains("hhc") || lower.contains("chm") {
            "chm"
        } else if lower.contains("wordperfect") || lower.contains("wpd") {
            "wordperfect"
        } else if lower.contains("quattro") || lower.contains("wb3") {
            "quattro-pro"
        } else {
            return None;
        };
        metadata.insert("legacy.kind", kind);
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings: Vec::new(),
            parser_chain: Vec::new(),
        })
    }
}

pub struct Rfc822MimeParser;
impl Parser for Rfc822MimeParser {
    fn name(&self) -> &'static str {
        "Rfc822MimeParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "message/rfc822" || media_type == "multipart/mixed"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "Rfc822MimeParser");
        if let Some(from) = extract_header(&content, "From:") {
            metadata.insert("mail.from", from);
        }
        if let Some(to) = extract_header(&content, "To:") {
            metadata.insert("mail.to", to);
        }
        if let Some(subject) = extract_header(&content, "Subject:") {
            metadata.insert("mail.subject", subject);
        }
        let charset = extract_mail_charset(&lower).unwrap_or_else(|| "utf-8".to_string());
        metadata.insert("mail.charset", charset.clone());
        let attachment_count = lower.matches("content-disposition: attachment").count()
            + lower.matches("filename=").count();
        metadata.insert("mail.attachment_count", attachment_count.to_string());
        let nested_count = lower.matches("message/rfc822").count().saturating_sub(1);
        metadata.insert("mail.nested_count", nested_count.to_string());
        let mut warnings = Vec::new();
        if charset == "unknown" {
            warnings.push("mail-invalid-charset".to_string());
        }
        if attachment_count > 64 {
            warnings.push("mail-attachment-limit".to_string());
        }
        Some(ParseOutcome {
            content: Some(extract_mail_body(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct MboxParser;
impl Parser for MboxParser {
    fn name(&self) -> &'static str {
        "MboxParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/mbox" || media_type == "application/x-mbox"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8_lossy(input);
        let messages = split_mbox_messages(&content);
        if messages.is_empty() {
            return None;
        }
        let mut metadata = Metadata::default();
        metadata.insert("parser", "MboxParser");
        metadata.insert("mail.message_count", messages.len().to_string());
        let mut total_attachments = 0usize;
        let mut total_nested = 0usize;
        let mut bodies = Vec::new();
        for msg in &messages {
            let lower = msg.to_ascii_lowercase();
            if let Some(from) = extract_header(msg, "From:") {
                metadata.insert("mail.from", from);
            }
            if let Some(subject) = extract_header(msg, "Subject:") {
                metadata.insert("mail.subject", subject);
            }
            total_attachments += lower.matches("content-disposition: attachment").count()
                + lower.matches("filename=").count();
            total_nested += lower.matches("message/rfc822").count();
            let body = extract_mail_body(msg);
            if !body.trim().is_empty() {
                bodies.push(body.trim().to_string());
            }
        }
        metadata.insert("mail.attachment_count", total_attachments.to_string());
        metadata.insert("mail.nested_count", total_nested.to_string());
        let mut warnings = Vec::new();
        if total_attachments > 256 {
            warnings.push("mail-attachment-limit".to_string());
        }
        Some(ParseOutcome {
            content: Some(bodies.join("\n\n")),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct OutlookMailboxParser;
impl Parser for OutlookMailboxParser {
    fn name(&self) -> &'static str {
        "OutlookMailboxParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/vnd.ms-outlook" || media_type == "application/x-tnef"
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "OutlookMailboxParser");
        let kind = if media_type == "application/x-tnef"
            || lower.contains("tnef")
            || lower.contains("winmail.dat")
        {
            "tnef"
        } else if lower.contains("__substg1.0_") || lower.contains("message class") {
            "msg"
        } else if lower.contains("!bdn") || lower.contains("pst") {
            "pst"
        } else {
            return None;
        };
        metadata.insert("mail.store_kind", kind);
        let attachment_count = if kind == "tnef" {
            lower.matches("attachrenddata").count() + lower.matches("attattach").count()
        } else {
            lower.matches("attach").count()
        };
        metadata.insert("mail.attachment_count", attachment_count.to_string());
        let message_count = if kind == "pst" {
            lower.matches("subject").count().max(1)
        } else {
            1
        };
        metadata.insert("mail.message_count", message_count.to_string());
        let mut warnings = Vec::new();
        if attachment_count > 512 {
            warnings.push("mail-attachment-limit".to_string());
        }
        if kind == "pst" && input.len() > 64 * 1024 * 1024 {
            warnings.push("mail-store-size-limit".to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&content)),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct ImageMetadataParser;
impl Parser for ImageMetadataParser {
    fn name(&self) -> &'static str {
        "ImageMetadataParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "image/jpeg"
                | "image/tiff"
                | "image/bpg"
                | "image/vnd.adobe.photoshop"
                | "image/webp"
                | "image/heif"
                | "image/icns"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let text = String::from_utf8_lossy(input);
        let lower = text.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "ImageMetadataParser");
        metadata.insert("image.mime", media_type);
        let format = if input.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "jpeg"
        } else if input.starts_with(b"II*\0") || input.starts_with(b"MM\0*") {
            "tiff"
        } else if input.starts_with(b"BPG\xFB") {
            "bpg"
        } else if input.starts_with(b"8BPS") {
            "psd"
        } else if input.len() > 12 && &input[0..4] == b"RIFF" && &input[8..12] == b"WEBP" {
            "webp"
        } else if lower.contains("ftypheic") || lower.contains("ftypmif1") {
            "heif"
        } else if input.starts_with(b"icns") {
            "icns"
        } else {
            "unknown"
        };
        metadata.insert("image.format", format);
        if lower.contains("exif") {
            metadata.insert("image.has_exif", "true");
        }
        if lower.contains("<x:xmpmeta") || lower.contains("http://ns.adobe.com/xap/1.0/") {
            metadata.insert("image.has_xmp", "true");
        }
        if lower.contains("iptc") || lower.contains("photoshop 3.0") {
            metadata.insert("image.has_iptc", "true");
        }
        let mut warnings = Vec::new();
        if format == "unknown" {
            warnings.push("image-corrupted-or-unknown".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct AudioMetadataParser;
impl Parser for AudioMetadataParser {
    fn name(&self) -> &'static str {
        "AudioMetadataParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(media_type, "audio/mpeg" | "audio/midi" | "audio/basic")
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let text = String::from_utf8_lossy(input);
        let lower = text.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "AudioMetadataParser");
        let format = if media_type == "audio/midi" || lower.contains("mthd") {
            "midi"
        } else if media_type == "audio/mpeg" || lower.contains("id3") || lower.contains("xffxfb") {
            "mp3"
        } else {
            "audio"
        };
        metadata.insert("audio.format", format);
        if let Some(v) = extract_tag_value(&text, "Title:") {
            metadata.insert("audio.title", v);
        }
        if let Some(v) = extract_tag_value(&text, "Artist:") {
            metadata.insert("audio.artist", v);
        }
        if let Some(v) = extract_tag_value(&text, "Album:") {
            metadata.insert("audio.album", v);
        }
        metadata.insert("audio.byte_length", input.len().to_string());
        let mut warnings = Vec::new();
        if media_type == "audio/mpeg" && !lower.contains("id3") && !lower.contains("xffxfb") {
            warnings.push("audio-bad-tag".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct VideoMetadataParser;
impl Parser for VideoMetadataParser {
    fn name(&self) -> &'static str {
        "VideoMetadataParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(media_type, "video/mp4" | "video/x-flv" | "video/*")
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "VideoMetadataParser");
        let format = if media_type == "video/mp4" || lower.contains("ftyp") || lower.contains("moov")
        {
            "mp4"
        } else if media_type == "video/x-flv" || input.starts_with(b"FLV") {
            "flv"
        } else {
            "video"
        };
        metadata.insert("video.format", format);
        metadata.insert("video.byte_length", input.len().to_string());
        let mut warnings = Vec::new();
        if input.len() > 16 * 1024 * 1024 {
            warnings.push("video-read-window-applied".to_string());
            metadata.insert("video.read_window_bytes", (4 * 1024 * 1024).to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct VisionBridgeParser;
impl Parser for VisionBridgeParser {
    fn name(&self) -> &'static str {
        "VisionBridgeParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type.starts_with("image/") || media_type.starts_with("video/")
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let content = String::from_utf8_lossy(input);
        let lower = content.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "VisionBridgeParser");
        let mut warnings = Vec::new();
        if lower.contains("caption:disable") || lower.contains("recognition:disable") {
            metadata.insert("vision.enabled", "false");
            warnings.push("vision-disabled".to_string());
            return Some(ParseOutcome {
                content: None,
                metadata,
                warnings,
                parser_chain: Vec::new(),
            });
        }
        metadata.insert("vision.enabled", "true");
        if lower.contains("caption:timeout") || lower.contains("recognition:timeout") {
            warnings.push("vision-timeout".to_string());
            return Some(ParseOutcome {
                content: None,
                metadata,
                warnings,
                parser_chain: Vec::new(),
            });
        }
        if lower.contains("caption:fail") || lower.contains("recognition:fail") {
            warnings.push("vision-failed-degraded".to_string());
            return Some(ParseOutcome {
                content: None,
                metadata,
                warnings,
                parser_chain: Vec::new(),
            });
        }
        metadata.insert("vision.caption", "generated");
        metadata.insert("vision.objects", "detected");
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct DatabaseTabularParser;
impl Parser for DatabaseTabularParser {
    fn name(&self) -> &'static str {
        "DatabaseTabularParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/x-dbf"
                | "application/vnd.sqlite3"
                | "application/x-msaccess"
                | "application/x-jdbc"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "DatabaseTabularParser");
        let kind = if media_type == "application/x-dbf" || lower.contains("dbf") {
            "dbf"
        } else if media_type == "application/vnd.sqlite3" || lower.contains("sqlite format 3") {
            "sqlite"
        } else if media_type == "application/x-msaccess"
            || lower.contains("standard jet db")
            || lower.contains("msysobjects")
        {
            "access"
        } else {
            "jdbc"
        };
        metadata.insert("db.kind", kind);
        metadata.insert("db.connection.enabled", "false");
        let mut warnings = vec!["db-connection-disabled-by-default".to_string()];
        if kind == "jdbc" && (lower.contains("jdbc:") || lower.contains("driverclass")) {
            warnings.push("db-jdbc-config-detected".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct ScienceDataParser;
impl Parser for ScienceDataParser {
    fn name(&self) -> &'static str {
        "ScienceDataParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/x-netcdf"
                | "application/x-hdf"
                | "application/x-grib"
                | "application/x-matlab-data"
                | "application/x-sas-data"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "ScienceDataParser");
        let kind = if media_type == "application/x-netcdf" || lower.contains("netcdf") {
            "netcdf"
        } else if media_type == "application/x-hdf" || lower.contains("hdf") {
            "hdf"
        } else if media_type == "application/x-grib" || lower.contains("grib") {
            "grib"
        } else if media_type == "application/x-matlab-data" || lower.contains("matlab") {
            "mat"
        } else {
            "sas"
        };
        metadata.insert("science.kind", kind);
        metadata.insert("science.native_feature", "optional");
        let mut warnings = Vec::new();
        if kind == "hdf" || kind == "grib" {
            warnings.push("science-native-dependency-optional".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct GeoEngineeringParser;
impl Parser for GeoEngineeringParser {
    fn name(&self) -> &'static str {
        "GeoEngineeringParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/x-gdal"
                | "application/acad"
                | "application/x-geodata"
                | "application/x-geographic-info"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "GeoEngineeringParser");
        let kind = if media_type == "application/x-gdal" || lower.contains("gdal") {
            "gdal"
        } else if media_type == "application/acad" || lower.contains("autocad") || lower.contains("dwg")
        {
            "dwg"
        } else if lower.contains("epsg:") || lower.contains("geometry") || lower.contains("geojson") {
            "geo"
        } else {
            "geographic-information"
        };
        metadata.insert("geo.kind", kind);
        metadata.insert("geo.native_feature", "optional");
        let mut warnings = Vec::new();
        if kind == "gdal" || kind == "dwg" {
            warnings.push("geo-native-dependency-optional".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct SpecialistFormatParser;
impl Parser for SpecialistFormatParser {
    fn name(&self) -> &'static str {
        "SpecialistFormatParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/x-isatab"
                | "application/x-grobid-tei"
                | "application/x-pooled-timeseries"
                | "application/x-pot"
                | "application/x-prt"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "SpecialistFormatParser");
        let kind = if media_type == "application/x-isatab" || lower.contains("investigation.txt") {
            "isa-tab"
        } else if media_type == "application/x-grobid-tei" || lower.contains("<tei") {
            "grobid-journal"
        } else if media_type == "application/x-pooled-timeseries" || lower.contains("timeseries") {
            "pooled-timeseries"
        } else if media_type == "application/x-pot" || lower.contains("msgid") {
            "pot"
        } else {
            "prt"
        };
        metadata.insert("special.kind", kind);
        let mut warnings = Vec::new();
        if kind == "grobid-journal" && lower.contains("service:disabled") {
            warnings.push("special-external-service-disabled".to_string());
        }
        if kind == "grobid-journal" && lower.contains("service:timeout") {
            warnings.push("special-external-service-timeout".to_string());
        }
        Some(ParseOutcome {
            content: Some(strip_html_tags(&String::from_utf8_lossy(input))),
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct CryptoSecurityParser;
impl Parser for CryptoSecurityParser {
    fn name(&self) -> &'static str {
        "CryptoSecurityParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(media_type, "application/pkcs7-mime" | "application/x-tsd" | "application/x-encrypted")
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "CryptoSecurityParser");
        let kind = if media_type == "application/pkcs7-mime" || lower.contains("pkcs7") {
            "pkcs7"
        } else if media_type == "application/x-tsd" || lower.contains("timestamp token") {
            "tsd"
        } else {
            "encrypted-document"
        };
        metadata.insert("crypto.kind", kind);
        metadata.insert(
            "crypto.provider",
            if lower.contains("provider:bc") { "bouncycastle" } else { "default" },
        );
        if lower.contains("perm:read-only") {
            metadata.insert("crypto.permission", "read-only");
        }
        let mut warnings = Vec::new();
        if lower.contains("password:wrong") {
            warnings.push("crypto-password-invalid".to_string());
        } else if lower.contains("password:ok") {
            metadata.insert("crypto.password_status", "ok");
        } else if kind == "encrypted-document" {
            warnings.push("crypto-password-required".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct BinaryFontParser;
impl Parser for BinaryFontParser {
    fn name(&self) -> &'static str {
        "BinaryFontParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        matches!(
            media_type,
            "application/java-vm"
                | "application/x-executable"
                | "application/x-font-afm"
                | "font/ttf"
        )
    }
    fn parse(&self, input: &[u8], media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "BinaryFontParser");
        let kind = if media_type == "application/java-vm" || input.starts_with(&[0xCA, 0xFE, 0xBA, 0xBE]) {
            "java-class"
        } else if media_type == "application/x-executable"
            || input.starts_with(b"\x7FELF")
            || input.starts_with(b"MZ")
        {
            "executable"
        } else if media_type == "application/x-font-afm" || lower.contains("startfontmetrics") {
            "afm"
        } else {
            "truetype"
        };
        metadata.insert("binary.kind", kind);
        let mut warnings = Vec::new();
        if kind == "executable" {
            warnings.push("binary-security-scan-limited".to_string());
        }
        if kind == "java-class" && input.len() > 10 * 1024 * 1024 {
            warnings.push("binary-class-size-limit".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct LanguageIdParser;
impl Parser for LanguageIdParser {
    fn name(&self) -> &'static str {
        "LanguageIdParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type.starts_with("text/") || media_type == "application/lang-detect"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        if input.is_empty() {
            return None;
        }
        let text = String::from_utf8_lossy(input);
        let trimmed = text.trim();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "LanguageIdParser");
        metadata.insert("lang.ngram.profile", "builtin-v1");
        let (lang, confidence) = detect_language_simple(trimmed);
        metadata.insert("lang.code", lang);
        metadata.insert("lang.confidence", format!("{confidence:.2}"));
        let mut warnings = Vec::new();
        if trimmed.chars().count() < 16 || confidence < 0.6 {
            warnings.push("lang-low-confidence".to_string());
        }
        Some(ParseOutcome {
            content: None,
            metadata,
            warnings,
            parser_chain: Vec::new(),
        })
    }
}

pub struct LanguageProviderParser;
impl Parser for LanguageProviderParser {
    fn name(&self) -> &'static str {
        "LanguageProviderParser"
    }
    fn supports(&self, media_type: &str) -> bool {
        media_type == "application/lang-provider"
    }
    fn parse(&self, input: &[u8], _media_type: &str) -> Option<ParseOutcome> {
        let text = String::from_utf8(input.to_vec()).ok()?;
        let lower = text.to_ascii_lowercase();
        let mut metadata = Metadata::default();
        metadata.insert("parser", "LanguageProviderParser");
        let provider = if lower.contains("provider=optimaize") {
            "optimaize"
        } else if lower.contains("provider=lingo24") {
            "lingo24"
        } else {
            "text"
        };
        metadata.insert("lang.provider", provider);
        let mut warnings = Vec::new();
        if lower.contains("enabled=false") {
            warnings.push("lang-provider-disabled".to_string());
        }
        if lower.contains("config=invalid") {
            warnings.push("lang-provider-config-invalid".to_string());
        }
        Some(ParseOutcome {
            content: None,
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

fn strip_rtf_control_words(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            while let Some(nc) = chars.peek() {
                if nc.is_ascii_alphabetic() || nc.is_ascii_digit() || *nc == '-' {
                    let _ = chars.next();
                } else {
                    break;
                }
            }
            continue;
        }
        if ch == '{' || ch == '}' {
            continue;
        }
        out.push(ch);
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_header(content: &str, key: &str) -> Option<String> {
    content
        .lines()
        .find(|l| l.starts_with(key))
        .map(|l| l[key.len()..].trim().to_string())
}

fn extract_tag_value(content: &str, key: &str) -> Option<String> {
    content
        .lines()
        .find(|l| l.starts_with(key))
        .map(|l| l[key.len()..].trim().to_string())
}

fn detect_language_simple(text: &str) -> (&'static str, f32) {
    if text.is_empty() {
        return ("und", 0.0);
    }
    let lower = text.to_ascii_lowercase();
    if lower.contains(" the ") || lower.contains(" and ") {
        return ("en", 0.86);
    }
    if lower.contains(" el ") || lower.contains(" la ") || lower.contains(" de ") {
        return ("es", 0.78);
    }
    if lower.contains(" le ") || lower.contains(" les ") || lower.contains(" des ") {
        return ("fr", 0.76);
    }
    if text.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c)) {
        return ("zh", 0.82);
    }
    ("und", 0.45)
}

fn extract_mail_charset(lower: &str) -> Option<String> {
    let idx = lower.find("charset=")?;
    let mut rest = &lower[idx + "charset=".len()..];
    rest = rest.trim_start();
    if let Some(s) = rest.strip_prefix('"').or_else(|| rest.strip_prefix('\'')) {
        rest = s;
    }
    let end = rest
        .find(|c: char| c == '"' || c == '\'' || c == ';' || c.is_whitespace())
        .unwrap_or(rest.len());
    let cs = rest[..end].trim();
    if cs.is_empty() {
        None
    } else if matches!(cs, "utf-8" | "utf8" | "iso-8859-1" | "gbk" | "us-ascii") {
        Some(cs.to_string())
    } else {
        Some("unknown".to_string())
    }
}

fn extract_mail_body(content: &str) -> String {
    if let Some((_, body)) = content.split_once("\r\n\r\n") {
        return body.to_string();
    }
    if let Some((_, body)) = content.split_once("\n\n") {
        return body.to_string();
    }
    content.to_string()
}

fn split_mbox_messages(content: &str) -> Vec<String> {
    let mut messages = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        if line.starts_with("From ") && !current.trim().is_empty() {
            messages.push(current.trim().to_string());
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        messages.push(current.trim().to_string());
    }
    messages
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

fn detect_package_kind(input: &[u8]) -> Option<&'static str> {
    if input.starts_with(b"PK\x03\x04") || input.starts_with(b"PK\x05\x06") {
        return Some("zip");
    }
    if input.starts_with(b"\x1F\x8B") {
        return Some("gzip");
    }
    if input.starts_with(b"BZh") {
        return Some("bzip2");
    }
    if input.starts_with(b"\xFD7zXZ\x00") {
        return Some("xz");
    }
    if input.starts_with(&[0x28, 0xB5, 0x2F, 0xFD]) {
        return Some("zstd");
    }
    if input.starts_with(b"7z\xBC\xAF\x27\x1C") {
        return Some("7z");
    }
    if input.starts_with(b"Rar!\x1A\x07\x00") || input.starts_with(b"Rar!\x1A\x07\x01\x00") {
        return Some("rar");
    }
    if input.len() > 262 && &input[257..262] == b"ustar" {
        return Some("tar");
    }
    None
}

fn estimate_archive_stats(input: &[u8]) -> (usize, usize) {
    let entry_markers = input
        .windows(8)
        .filter(|w| *w == b"[[FILE:]]" || *w == b"[[DIR:]]")
        .count();
    let entry_count = entry_markers.max(1);
    let inflated = input.len().saturating_mul(8).saturating_add(entry_count * 64);
    (entry_count, inflated)
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
        CompositeParser, DerivedTextParser, FeedParser, HtmlParser, MetadataOnlyParser, Parser,
        EpubParser, IworkParser, LightweightSpecializedParser, OdfParser, OoxmlParser, PackageParser,
        LegacyDocParser, MboxParser, MsSpecialParser, OleLegacyParser, OutlookMailboxParser,
        AudioMetadataParser, ImageMetadataParser, PdfParser, Rfc822MimeParser, RtfParser,
        DatabaseTabularParser, VideoMetadataParser, VisionBridgeParser,
        BinaryFontParser, CryptoSecurityParser, GeoEngineeringParser, ScienceDataParser,
        LanguageIdParser, LanguageProviderParser, SpecialistFormatParser,
        SourceCodeParser, StringsParser, TextAndCsvParser, TxtParser, XmlParser,
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

    #[test]
    fn derived_text_parser_detects_min_formats() {
        let p = DerivedTextParser;
        let xliff = p.parse(b"<xliff version=\"1.2\"></xliff>", "application/x-xliff+xml").expect("x");
        assert_eq!(
            xliff
                .metadata
                .values("derived.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("xliff")
        );
        let dif = p.parse(b"TABLE\n0,1\nVECTORS", "text/x-dif").expect("d");
        assert_eq!(
            dif.metadata
                .values("derived.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("dif")
        );
        let envi = p
            .parse(b"ENVI\nsamples = 128", "text/x-envi-header")
            .expect("e");
        assert_eq!(
            envi.metadata
                .values("derived.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("envi-header")
        );
    }

    #[test]
    fn lightweight_specialized_parser_profiles_match() {
        let p = LightweightSpecializedParser;
        let plist = p
            .parse(b"<?xml version='1.0'?><plist><dict/></plist>", "application/x-plist")
            .expect("plist");
        assert_eq!(
            plist
                .metadata
                .values("lightweight.profile")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("PList")
        );
        let fb = p
            .parse(b"<FictionBook><body/></FictionBook>", "application/xml")
            .expect("fb");
        assert_eq!(
            fb.metadata
                .values("lightweight.profile")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("FictionBook")
        );
        let dc = p.parse(b"<dc:title>x</dc:title>", "application/xml").expect("dc");
        assert_eq!(
            dc.metadata
                .values("lightweight.profile")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("DcXML")
        );
    }

    #[test]
    fn package_parser_detects_kinds_and_limits() {
        let p = PackageParser;
        let zip = p.parse(b"PK\x03\x04....", "application/zip").expect("zip");
        assert_eq!(
            zip.metadata
                .values("package.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("zip")
        );
        let rar = p
            .parse(b"Rar!\x1A\x07\x00....", "application/vnd.rar")
            .expect("rar");
        assert_eq!(
            rar.metadata
                .values("package.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("rar")
        );
    }

    #[test]
    fn ooxml_parser_extracts_kind_relationships_props_and_embeds() {
        let p = OoxmlParser;
        let out = p
            .parse(
                b"PK\x03\x04...word/document.xml..._rels/.rels...docProps/core.xml...embeddings/oleObject1.bin",
                "application/x-tika-ooxml",
            )
            .expect("ooxml");
        assert_eq!(
            out.metadata
                .values("ooxml.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("docx")
        );
        assert_eq!(
            out.metadata
                .values("ooxml.relationships")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("ooxml.core_props")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn odf_parser_extracts_kind_manifest_and_meta() {
        let p = OdfParser;
        let out = p
            .parse(
                b"PK\x03\x04...mimetypeapplication/vnd.oasis.opendocument.text...META-INF/manifest.xml...meta.xml",
                "application/vnd.oasis.opendocument",
            )
            .expect("odf");
        assert_eq!(
            out.metadata
                .values("odf.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("odt")
        );
        assert_eq!(
            out.metadata
                .values("odf.manifest")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("odf.meta")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn epub_parser_extracts_spine_metadata_and_embeds() {
        let p = EpubParser;
        let out = p
            .parse(
                b"PK\x03\x04...META-INF/container.xml...<spine><itemref/></spine>...<dc:title>T</dc:title>...images/a.jpg",
                "application/epub+zip",
            )
            .expect("epub");
        assert_eq!(
            out.metadata
                .values("epub.container")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("epub.spine")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("epub.metadata")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn iwork_parser_detects_pages_numbers_keynote() {
        let p = IworkParser;
        let pages = p
            .parse(
                b"PK\x03\x04...Index/Document.iwa...Pages",
                "application/x-iwork-package",
            )
            .expect("pages");
        assert_eq!(
            pages
                .metadata
                .values("iwork.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("pages")
        );
        let numbers = p
            .parse(
                b"PK\x03\x04...Index/Tables/Table.iwa...Numbers",
                "application/x-iwork-package",
            )
            .expect("numbers");
        assert_eq!(
            numbers
                .metadata
                .values("iwork.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("numbers")
        );
        let keynote = p
            .parse(
                b"PK\x03\x04...Index/Metadata.iwa...Keynote",
                "application/x-iwork-package",
            )
            .expect("keynote");
        assert_eq!(
            keynote
                .metadata
                .values("iwork.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("keynote")
        );
    }

    #[test]
    fn pdf_parser_extracts_metadata_and_warnings() {
        let p = PdfParser;
        let out = p
            .parse(
                b"%PDF-1.7\n/Encrypt true /EmbeddedFile /EmbeddedFile OCR:IMAGE preflight:error",
                "application/pdf",
            )
            .expect("pdf");
        assert_eq!(
            out.metadata
                .values("pdf.version")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("1.7")
        );
        assert_eq!(
            out.metadata
                .values("pdf.encrypted")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("pdf.attachment_count")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("2")
        );
        assert!(out.warnings.iter().any(|w| w == "pdf-ocr-hook-suggested"));
        assert!(out.warnings.iter().any(|w| w == "pdf-preflight-warning"));
    }

    #[test]
    fn ole_legacy_parser_detects_doc_xls_ppt_and_security_warnings() {
        let p = OleLegacyParser;
        let doc = p
            .parse(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1...WordDocument...VBA...OLE10Native",
                "application/x-tika-msoffice",
            )
            .expect("doc");
        assert_eq!(
            doc.metadata
                .values("ole.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("doc")
        );
        assert!(doc.warnings.iter().any(|w| w == "ole-macro-present"));
        assert!(doc.warnings.iter().any(|w| w == "ole-embedded-object"));

        let xls = p
            .parse(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1...Workbook...",
                "application/vnd.ms-excel",
            )
            .expect("xls");
        assert_eq!(
            xls.metadata
                .values("ole.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("xls")
        );

        let ppt = p
            .parse(
                b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1...PowerPoint Document...",
                "application/vnd.ms-powerpoint",
            )
            .expect("ppt");
        assert_eq!(
            ppt.metadata
                .values("ole.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("ppt")
        );
    }

    #[test]
    fn ms_special_parser_detects_onenote_access_tnef_emf_wmf_msg_pst() {
        let p = MsSpecialParser;
        let one = p.parse(b"...OneNote section...", "application/x-tika-msoffice").expect("one");
        assert_eq!(
            one.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("onenote")
        );
        let access = p
            .parse(b"...Standard Jet DB...MSysObjects...", "application/x-tika-msoffice")
            .expect("access");
        assert_eq!(
            access.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("access")
        );
        let tnef = p.parse(b"...winmail.dat...TNEF...", "application/x-tnef").expect("tnef");
        assert_eq!(
            tnef.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("tnef")
        );
        let emf = p.parse(b"... EMF+ ...", "image/emf").expect("emf");
        assert_eq!(
            emf.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("emf")
        );
        let wmf = p.parse(b"...metafile...WMF...", "image/wmf").expect("wmf");
        assert_eq!(
            wmf.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("wmf")
        );
        let msg = p
            .parse(b"...__substg1.0_...attach...", "application/vnd.ms-outlook")
            .expect("msg");
        assert_eq!(
            msg.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("msg")
        );
        let pst = p
            .parse(b"...!BDN...attach...", "application/vnd.ms-outlook")
            .expect("pst");
        assert_eq!(
            pst.metadata.values("ms.kind").and_then(|v| v.first()).map(String::as_str),
            Some("pst")
        );
    }

    #[test]
    fn rtf_parser_extracts_text_and_object_metadata() {
        let p = RtfParser;
        let out = p
            .parse(
                br"{\rtf1\ansi hello {\object\objdata 0102} world}",
                "application/rtf",
            )
            .expect("rtf");
        assert!(out.content.as_deref().unwrap_or("").contains("hello"));
        assert_eq!(
            out.metadata
                .values("rtf.object_count")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("2")
        );
    }

    #[test]
    fn legacy_doc_parser_detects_hwp_chm_wordperfect_quattro() {
        let p = LegacyDocParser;
        let hwp = p
            .parse(b"...HWP Document...Hangul Word Processor...", "application/x-hwp")
            .expect("hwp");
        assert_eq!(
            hwp.metadata
                .values("legacy.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("hwp")
        );
        let chm = p
            .parse(b"ITSF...chm...", "application/vnd.ms-htmlhelp")
            .expect("chm");
        assert_eq!(
            chm.metadata
                .values("legacy.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("chm")
        );
        let wpd = p
            .parse(b"...WordPerfect...WPD...", "application/wordperfect")
            .expect("wpd");
        assert_eq!(
            wpd.metadata
                .values("legacy.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("wordperfect")
        );
        let qpro = p
            .parse(b"...Quattro...WB3...", "application/x-quattro-pro")
            .expect("qpro");
        assert_eq!(
            qpro.metadata
                .values("legacy.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("quattro-pro")
        );
    }

    #[test]
    fn rfc822_mime_parser_extracts_headers_charset_and_attachments() {
        let p = Rfc822MimeParser;
        let out = p
            .parse(
                b"From: a@example.com\nTo: b@example.com\nSubject: hello\nContent-Type: multipart/mixed; charset=\"utf-8\"\n\nbody\nContent-Disposition: attachment; filename=x.txt",
                "message/rfc822",
            )
            .expect("mail");
        assert_eq!(
            out.metadata.values("mail.from").and_then(|v| v.first()).map(String::as_str),
            Some("a@example.com")
        );
        assert_eq!(
            out.metadata.values("mail.charset").and_then(|v| v.first()).map(String::as_str),
            Some("utf-8")
        );
        assert_eq!(
            out.metadata
                .values("mail.attachment_count")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("2")
        );
        assert!(out.content.as_deref().unwrap_or("").contains("body"));
    }

    #[test]
    fn rfc822_parser_warns_on_bad_charset() {
        let p = Rfc822MimeParser;
        let out = p
            .parse(
                b"Content-Type: text/plain; charset=not-a-real-charset\n\nx",
                "message/rfc822",
            )
            .expect("mail");
        assert!(out.warnings.iter().any(|w| w == "mail-invalid-charset"));
    }

    #[test]
    fn mbox_parser_extracts_multiple_messages() {
        let p = MboxParser;
        let out = p
            .parse(
                b"From sender1@example.com Sat Jan 01 00:00:00 2022\nFrom: sender1@example.com\nSubject: one\n\nbody one\nFrom sender2@example.com Sat Jan 01 00:00:01 2022\nFrom: sender2@example.com\nSubject: two\nContent-Type: message/rfc822\n\nbody two",
                "application/mbox",
            )
            .expect("mbox");
        assert_eq!(
            out.metadata
                .values("mail.message_count")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("2")
        );
        assert_eq!(
            out.metadata
                .values("mail.nested_count")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("1")
        );
        assert!(out.content.as_deref().unwrap_or("").contains("body one"));
        assert!(out.content.as_deref().unwrap_or("").contains("body two"));
    }

    #[test]
    fn mbox_parser_warns_when_attachment_limit_exceeded() {
        let p = MboxParser;
        let mut mail = String::from("From sender@example.com\nFrom: sender@example.com\n\n");
        for _ in 0..257 {
            mail.push_str("Content-Disposition: attachment; filename=a.bin\n");
        }
        let out = p.parse(mail.as_bytes(), "application/x-mbox").expect("mbox");
        assert!(out.warnings.iter().any(|w| w == "mail-attachment-limit"));
    }

    #[test]
    fn outlook_mailbox_parser_detects_msg_pst_tnef() {
        let p = OutlookMailboxParser;
        let msg = p
            .parse(
                b"__substg1.0_...Message Class: IPM.Note...attach...",
                "application/vnd.ms-outlook",
            )
            .expect("msg");
        assert_eq!(
            msg.metadata
                .values("mail.store_kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("msg")
        );
        let pst = p
            .parse(b"...!BDN...pst...Subject: a\nSubject: b\nattach...", "application/vnd.ms-outlook")
            .expect("pst");
        assert_eq!(
            pst.metadata
                .values("mail.store_kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("pst")
        );
        let tnef = p
            .parse(
                b"...winmail.dat...TNEF...AttachRendData...attAttach...",
                "application/x-tnef",
            )
            .expect("tnef");
        assert_eq!(
            tnef.metadata
                .values("mail.store_kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("tnef")
        );
    }

    #[test]
    fn image_metadata_parser_extracts_format_and_embedded_metadata_flags() {
        let p = ImageMetadataParser;
        let out = p
            .parse(
                b"\xFF\xD8\xFF....EXIF....<x:xmpmeta>..IPTC..",
                "image/jpeg",
            )
            .expect("jpeg");
        assert_eq!(
            out.metadata
                .values("image.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("jpeg")
        );
        assert_eq!(
            out.metadata
                .values("image.has_exif")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("image.has_xmp")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(
            out.metadata
                .values("image.has_iptc")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn image_metadata_parser_warns_on_unknown_signature() {
        let p = ImageMetadataParser;
        let out = p.parse(b"not-an-image", "image/webp").expect("image");
        assert!(out
            .warnings
            .iter()
            .any(|w| w == "image-corrupted-or-unknown"));
    }

    #[test]
    fn audio_metadata_parser_extracts_mp3_tags() {
        let p = AudioMetadataParser;
        let out = p
            .parse(
                b"ID3\nTitle: Song A\nArtist: Artist A\nAlbum: Album A\n",
                "audio/mpeg",
            )
            .expect("audio");
        assert_eq!(
            out.metadata
                .values("audio.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("mp3")
        );
        assert_eq!(
            out.metadata
                .values("audio.title")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("Song A")
        );
    }

    #[test]
    fn audio_metadata_parser_warns_on_bad_mp3_tag() {
        let p = AudioMetadataParser;
        let out = p.parse(b"raw-audio-stream", "audio/mpeg").expect("audio");
        assert!(out.warnings.iter().any(|w| w == "audio-bad-tag"));
    }

    #[test]
    fn video_metadata_parser_detects_mp4_and_flv() {
        let p = VideoMetadataParser;
        let mp4 = p.parse(b"...ftyp...moov...", "video/mp4").expect("mp4");
        assert_eq!(
            mp4.metadata
                .values("video.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("mp4")
        );
        let flv = p.parse(b"FLV\x01\x05", "video/x-flv").expect("flv");
        assert_eq!(
            flv.metadata
                .values("video.format")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("flv")
        );
    }

    #[test]
    fn video_metadata_parser_applies_read_window_warning_on_large_inputs() {
        let p = VideoMetadataParser;
        let huge = vec![b'a'; 17 * 1024 * 1024];
        let out = p.parse(&huge, "video/mp4").expect("video");
        assert!(out
            .warnings
            .iter()
            .any(|w| w == "video-read-window-applied"));
    }

    #[test]
    fn vision_bridge_parser_handles_disabled_timeout_and_failure_degrade() {
        let p = VisionBridgeParser;
        let disabled = p
            .parse(b"caption:disable", "image/jpeg")
            .expect("disabled");
        assert!(disabled.warnings.iter().any(|w| w == "vision-disabled"));

        let timeout = p.parse(b"caption:timeout", "video/mp4").expect("timeout");
        assert!(timeout.warnings.iter().any(|w| w == "vision-timeout"));

        let failed = p.parse(b"recognition:fail", "image/webp").expect("fail");
        assert!(failed
            .warnings
            .iter()
            .any(|w| w == "vision-failed-degraded"));
    }

    #[test]
    fn database_tabular_parser_detects_kinds_and_connection_policy() {
        let p = DatabaseTabularParser;
        let sqlite = p
            .parse(b"SQLite format 3\0....", "application/vnd.sqlite3")
            .expect("sqlite");
        assert_eq!(
            sqlite
                .metadata
                .values("db.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("sqlite")
        );
        let jdbc = p
            .parse(
                b"jdbc:mysql://localhost:3306/demo DriverClass=com.mysql.Driver",
                "application/x-jdbc",
            )
            .expect("jdbc");
        assert!(jdbc
            .warnings
            .iter()
            .any(|w| w == "db-connection-disabled-by-default"));
        assert!(jdbc
            .warnings
            .iter()
            .any(|w| w == "db-jdbc-config-detected"));
    }

    #[test]
    fn science_data_parser_detects_kinds_and_native_feature_flags() {
        let p = ScienceDataParser;
        let netcdf = p
            .parse(b"CDF...netcdf...", "application/x-netcdf")
            .expect("netcdf");
        assert_eq!(
            netcdf
                .metadata
                .values("science.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("netcdf")
        );
        let hdf = p.parse(b"HDF5...", "application/x-hdf").expect("hdf");
        assert!(hdf
            .warnings
            .iter()
            .any(|w| w == "science-native-dependency-optional"));
    }

    #[test]
    fn geo_engineering_parser_detects_gdal_dwg_and_geo() {
        let p = GeoEngineeringParser;
        let gdal = p.parse(b"GDAL dataset", "application/x-gdal").expect("gdal");
        assert_eq!(
            gdal.metadata
                .values("geo.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("gdal")
        );
        let dwg = p.parse(b"AutoCAD DWG", "application/acad").expect("dwg");
        assert!(dwg
            .warnings
            .iter()
            .any(|w| w == "geo-native-dependency-optional"));
        let geo = p
            .parse(b"{\"type\":\"Feature\",\"crs\":\"EPSG:4326\"}", "application/x-geodata")
            .expect("geo");
        assert_eq!(
            geo.metadata
                .values("geo.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("geo")
        );
    }

    #[test]
    fn specialist_format_parser_detects_kind_and_service_degrade() {
        let p = SpecialistFormatParser;
        let isa = p
            .parse(b"investigation.txt study.txt assay.txt", "application/x-isatab")
            .expect("isa");
        assert_eq!(
            isa.metadata
                .values("special.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("isa-tab")
        );
        let grobid = p
            .parse(
                b"<TEI>service:disabled service:timeout</TEI>",
                "application/x-grobid-tei",
            )
            .expect("grobid");
        assert!(grobid
            .warnings
            .iter()
            .any(|w| w == "special-external-service-disabled"));
        assert!(grobid
            .warnings
            .iter()
            .any(|w| w == "special-external-service-timeout"));
    }

    #[test]
    fn crypto_security_parser_handles_password_and_permission_states() {
        let p = CryptoSecurityParser;
        let ok = p
            .parse(b"pkcs7 provider:bc password:ok perm:read-only", "application/pkcs7-mime")
            .expect("ok");
        assert_eq!(
            ok.metadata
                .values("crypto.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("pkcs7")
        );
        assert_eq!(
            ok.metadata
                .values("crypto.permission")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("read-only")
        );
        let bad = p
            .parse(b"encrypted blob password:wrong", "application/x-encrypted")
            .expect("bad");
        assert!(bad
            .warnings
            .iter()
            .any(|w| w == "crypto-password-invalid"));
    }

    #[test]
    fn binary_font_parser_detects_class_exec_and_fonts() {
        let p = BinaryFontParser;
        let class = p
            .parse(&[0xCA, 0xFE, 0xBA, 0xBE, 0x00], "application/java-vm")
            .expect("class");
        assert_eq!(
            class
                .metadata
                .values("binary.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("java-class")
        );
        let exe = p.parse(b"MZ....", "application/x-executable").expect("exe");
        assert!(exe
            .warnings
            .iter()
            .any(|w| w == "binary-security-scan-limited"));
        let afm = p
            .parse(b"StartFontMetrics 4.1", "application/x-font-afm")
            .expect("afm");
        assert_eq!(
            afm.metadata
                .values("binary.kind")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("afm")
        );
    }

    #[test]
    fn language_id_parser_detects_language_and_low_confidence_paths() {
        let p = LanguageIdParser;
        let en = p
            .parse(b"This is the document and this is the second line.", "text/plain")
            .expect("en");
        assert_eq!(
            en.metadata
                .values("lang.code")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("en")
        );
        let short = p.parse(b"hi", "text/plain").expect("short");
        assert!(short
            .warnings
            .iter()
            .any(|w| w == "lang-low-confidence"));
    }

    #[test]
    fn language_provider_parser_handles_switch_disable_and_bad_config() {
        let p = LanguageProviderParser;
        let opt = p
            .parse(b"provider=optimaize enabled=true", "application/lang-provider")
            .expect("opt");
        assert_eq!(
            opt.metadata
                .values("lang.provider")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("optimaize")
        );
        let bad = p
            .parse(
                b"provider=lingo24 enabled=false config=invalid",
                "application/lang-provider",
            )
            .expect("bad");
        assert!(bad
            .warnings
            .iter()
            .any(|w| w == "lang-provider-disabled"));
        assert!(bad
            .warnings
            .iter()
            .any(|w| w == "lang-provider-config-invalid"));
    }
}
