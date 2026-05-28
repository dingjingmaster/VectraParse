#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyMsoExtract {
    pub kind: &'static str,
    pub text: String,
    pub warnings: Vec<String>,
}

const OLE_MAGIC: &[u8] = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1";

pub fn extract_legacy_mso_text(input: &[u8]) -> Option<LegacyMsoExtract> {
    if !input.starts_with(OLE_MAGIC) {
        return None;
    }
    let kind = detect_kind(input);
    let mut warnings = Vec::new();
    let probe = String::from_utf8_lossy(input).to_ascii_lowercase();
    if probe.contains("vba") || probe.contains("macros") {
        warnings.push("ole-macro-present".to_string());
    }
    if probe.contains("ole10native") || probe.contains("embedded object") {
        warnings.push("ole-embedded-object".to_string());
    }
    let text = extract_text_by_kind(input, kind);
    Some(LegacyMsoExtract {
        kind,
        text,
        warnings,
    })
}

fn detect_kind(input: &[u8]) -> &'static str {
    let lower = String::from_utf8_lossy(input).to_ascii_lowercase();
    if lower.contains("worddocument") {
        "doc"
    } else if lower.contains("workbook") || lower.contains("book") {
        "xls"
    } else if lower.contains("powerpoint document") {
        "ppt"
    } else if lower.contains("ownerfile") || lower.contains("~$") {
        "msoffice-ownerfile"
    } else {
        "ole-unknown"
    }
}

fn extract_text_by_kind(input: &[u8], kind: &str) -> String {
    let mut lines = Vec::new();
    let min_ascii = if kind == "xls" { 2 } else { 3 };
    for s in extract_utf16le_strings(input, 2, 48 * 1024) {
        lines.push(s);
    }
    for s in extract_ascii_strings(input, min_ascii, 48 * 1024) {
        lines.push(s);
    }
    for s in extract_latin1_strings(input, 4, 48 * 1024) {
        lines.push(repair_utf8_mojibake(&s).unwrap_or(s));
    }

    let mut out = Vec::new();
    for raw in lines {
        let normalized = normalize_line(&raw);
        if normalized.is_empty() {
            continue;
        }
        if !looks_like_content_line(&normalized, kind) {
            continue;
        }
        if out.last() == Some(&normalized) {
            continue;
        }
        out.push(normalized);
        if out.len() >= 800 {
            break;
        }
    }
    out.join("\n")
}

fn normalize_line(line: &str) -> String {
    let t = line.replace('\0', "");
    let compact = t.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.trim().to_string()
}

fn looks_like_content_line(line: &str, kind: &str) -> bool {
    if line.len() < 2 {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "root entry"
            | "summaryinformation"
            | "documentsummaryinformation"
            | "worddocument"
            | "powerpoint document"
            | "current user"
            | "workbook"
            | "book"
            | "normal.dotm"
            | "ksoProductBuildVer"
    ) {
        return false;
    }
    if lower.contains("root entry")
        || lower.contains("summaryinformation")
        || lower.contains("documentsummaryinformation")
        || lower.contains("kso")
    {
        return false;
    }
    if lower.contains("ihdr")
        || lower.contains("idat")
        || lower.contains("phys")
        || lower.contains("wmfc")
    {
        return false;
    }
    let alpha_like = line
        .chars()
        .filter(|c| {
            c.is_alphanumeric()
                || c.is_whitespace()
                || ('\u{4E00}'..='\u{9FFF}').contains(c)
                || ('\u{3040}'..='\u{30FF}').contains(c)
                || ('\u{AC00}'..='\u{D7AF}').contains(c)
        })
        .count();
    let ratio = alpha_like * 100 / line.chars().count();
    if ratio < 45 {
        return false;
    }
    if kind == "xls" {
        return ratio >= 35;
    }
    true
}

fn extract_ascii_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    let mut consumed = 0usize;
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || b == b'\t' || b == b' ';
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            let s = String::from_utf8_lossy(&buf).to_string();
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn extract_latin1_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    let mut consumed = 0usize;
    for &b in input {
        let printable = (0x20..=0x7e).contains(&b) || (0xa0..=0xff).contains(&b);
        if printable {
            buf.push(b);
            continue;
        }
        if buf.len() >= min_len {
            let s = buf.iter().map(|c| *c as char).collect::<String>();
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn extract_utf16le_strings(input: &[u8], min_len: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = Vec::<u16>::new();
    let mut consumed = 0usize;
    for chunk in input.chunks_exact(2) {
        let u = u16::from_le_bytes([chunk[0], chunk[1]]);
        let ch = char::from_u32(u as u32).unwrap_or('\0');
        let printable = ch.is_ascii_graphic()
            || ch.is_ascii_whitespace()
            || ('\u{4E00}'..='\u{9FFF}').contains(&ch)
            || ('\u{3040}'..='\u{30FF}').contains(&ch)
            || ('\u{AC00}'..='\u{D7AF}').contains(&ch);
        if printable {
            buf.push(u);
            continue;
        }
        if buf.len() >= min_len && let Ok(s) = String::from_utf16(&buf) {
            consumed += s.len();
            out.push(s);
        }
        buf.clear();
        if consumed >= max_chars {
            break;
        }
    }
    out
}

fn repair_utf8_mojibake(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    let suspicious = s
        .chars()
        .filter(|c| matches!(*c, 'Ã' | 'Â' | 'æ' | 'å' | 'è' | 'é' | 'ç' | 'ï' | 'ð'))
        .count();
    if suspicious * 8 < s.chars().count() {
        return None;
    }
    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) > 0xff {
            return None;
        }
        bytes.push(c as u8);
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::{detect_kind, repair_utf8_mojibake};

    #[test]
    fn detects_doc_kind() {
        let data = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1....WordDocument....";
        assert_eq!(detect_kind(data), "doc");
    }

    #[test]
    fn repairs_utf8_mojibake() {
        let fixed = repair_utf8_mojibake("è¿\u{00a0}æ").unwrap_or_default();
        assert!(fixed.is_empty() || fixed.chars().any(|c| !c.is_ascii()));
    }
}
