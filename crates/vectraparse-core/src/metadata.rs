use std::collections::BTreeMap;

pub const NS_ACCESS_PERMISSIONS: &str = "AccessPermissions";
pub const NS_OFFICE: &str = "Office";
pub const NS_PDF: &str = "PDF";
pub const NS_XMP: &str = "XMP";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Metadata {
    inner: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    InvalidJson(String),
}

impl Metadata {
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.inner.entry(key.into()).or_default().push(value.into());
    }

    pub fn values(&self, key: &str) -> Option<&[String]> {
        self.inner.get(key).map(Vec::as_slice)
    }

    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (idx, (key, vals)) in self.inner.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(&escape_json(key));
            out.push_str("\":[");
            for (v_idx, val) in vals.iter().enumerate() {
                if v_idx > 0 {
                    out.push(',');
                }
                out.push('"');
                out.push_str(&escape_json(val));
                out.push('"');
            }
            out.push(']');
        }
        out.push('}');
        out
    }

    pub fn from_json(input: &str) -> Result<Self, MetadataError> {
        let trimmed = input.trim();
        if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
            return Err(MetadataError::InvalidJson(trimmed.to_string()));
        }
        let mut md = Metadata::default();
        let body = &trimmed[1..trimmed.len() - 1];
        if body.trim().is_empty() {
            return Ok(md);
        }
        for pair in split_top_level(body, ',') {
            let (raw_key, raw_values) =
                split_top_level_once(pair, ':').ok_or_else(|| MetadataError::InvalidJson(pair.to_string()))?;
            let key = unquote(raw_key.trim())?;
            let values_block = raw_values.trim();
            if !(values_block.starts_with('[') && values_block.ends_with(']')) {
                return Err(MetadataError::InvalidJson(values_block.to_string()));
            }
            let values_body = &values_block[1..values_block.len() - 1];
            if values_body.trim().is_empty() {
                md.inner.insert(key, Vec::new());
                continue;
            }
            let mut vals = Vec::new();
            for raw_value in split_top_level(values_body, ',') {
                vals.push(unquote(raw_value.trim())?);
            }
            md.inner.insert(key, vals);
        }
        Ok(md)
    }
}

fn split_top_level(input: &str, sep: char) -> Vec<&str> {
    let mut items = Vec::new();
    let mut start = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            c if c == sep && depth == 0 => {
                items.push(input[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    items.push(input[start..].trim());
    items
}

fn split_top_level_once(input: &str, sep: char) -> Option<(&str, &str)> {
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for (idx, ch) in input.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            c if c == sep && depth == 0 => {
                let left = input[..idx].trim();
                let right = input[idx + ch.len_utf8()..].trim();
                return Some((left, right));
            }
            _ => {}
        }
    }
    None
}

fn unquote(raw: &str) -> Result<String, MetadataError> {
    if raw.len() < 2 || !raw.starts_with('"') || !raw.ends_with('"') {
        return Err(MetadataError::InvalidJson(raw.to_string()));
    }
    let mut out = String::new();
    let mut chars = raw[1..raw.len() - 1].chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let next = chars
                .next()
                .ok_or_else(|| MetadataError::InvalidJson(raw.to_string()))?;
            match next {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                't' => out.push('\t'),
                _ => return Err(MetadataError::InvalidJson(raw.to_string())),
            }
            continue;
        }
        out.push(ch);
    }
    Ok(out)
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::{
        Metadata, MetadataError, NS_ACCESS_PERMISSIONS, NS_OFFICE, NS_PDF, NS_XMP,
    };

    #[test]
    fn metadata_round_trip_json() {
        let mut md = Metadata::default();
        md.insert(format!("{NS_PDF}:encrypted"), "true");
        md.insert(format!("{NS_PDF}:pages"), "12");
        md.insert(format!("{NS_XMP}:CreatorTool"), "Writer");
        md.insert(format!("{NS_OFFICE}:author"), "alice");
        md.insert(format!("{NS_ACCESS_PERMISSIONS}:can_print"), "false");
        let json = md.to_json();
        let parsed = Metadata::from_json(&json).expect("must parse");
        assert_eq!(md, parsed);
    }

    #[test]
    fn metadata_reject_invalid_json() {
        let err = Metadata::from_json("not json").expect_err("must fail");
        assert_eq!(err, MetadataError::InvalidJson("not json".to_string()));
    }
}
