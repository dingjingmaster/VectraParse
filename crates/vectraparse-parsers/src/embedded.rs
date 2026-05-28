use vectraparse_core::metadata::Metadata;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedNode {
    pub path: String,
    pub content: Vec<u8>,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedLimits {
    pub max_depth: usize,
    pub max_files: usize,
}

impl Default for EmbeddedLimits {
    fn default() -> Self {
        Self {
            max_depth: 5,
            max_files: 128,
        }
    }
}

pub fn collect_embedded(
    root_bytes: &[u8],
    limits: &EmbeddedLimits,
) -> (Vec<EmbeddedNode>, Vec<String>) {
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    collect_rec(
        root_bytes,
        "root",
        0,
        limits,
        &mut out,
        &mut warnings,
        &Metadata::default(),
    );
    (out, warnings)
}

fn collect_rec(
    bytes: &[u8],
    path: &str,
    depth: usize,
    limits: &EmbeddedLimits,
    out: &mut Vec<EmbeddedNode>,
    warnings: &mut Vec<String>,
    parent_md: &Metadata,
) {
    if depth >= limits.max_depth {
        warnings.push(format!("embedded-depth-limit:{path}"));
        return;
    }
    if out.len() >= limits.max_files {
        warnings.push(format!("embedded-file-limit:{path}"));
        return;
    }

    let mut idx = 0usize;
    while let Some(start_rel) = find_bytes(&bytes[idx..], b"[[EMBED:") {
        let start = idx + start_rel;
        let name_start = start + "[[EMBED:".len();
        let name_end_rel = match find_bytes(&bytes[name_start..], b"]]") {
            Some(v) => v,
            None => {
                warnings.push(format!("embedded-malformed-name:{path}"));
                return;
            }
        };
        let name_end = name_start + name_end_rel;
        let name = match std::str::from_utf8(&bytes[name_start..name_end]) {
            Ok(v) => v,
            Err(_) => {
                warnings.push(format!("embedded-invalid-name-utf8:{path}"));
                return;
            }
        };
        let content_start = name_end + 2;
        let end_rel = match find_matching_embed_end(&bytes[content_start..]) {
            Some(v) => v,
            None => {
                warnings.push(format!("embedded-missing-close:{path}/{name}"));
                return;
            }
        };
        let content_end = content_start + end_rel;
        let child_content = bytes[content_start..content_end].to_vec();
        let child_path = format!("{path}/{name}");
        let mut md = parent_md.clone();
        md.insert("embedded.path", child_path.clone());
        md.insert("embedded.depth", depth.to_string());

        out.push(EmbeddedNode {
            path: child_path.clone(),
            content: child_content.clone(),
            metadata: md.clone(),
        });
        collect_rec(
            &child_content,
            &child_path,
            depth + 1,
            limits,
            out,
            warnings,
            &md,
        );
        idx = content_end + "[[/EMBED]]".len();
        if out.len() >= limits.max_files {
            warnings.push(format!("embedded-file-limit:{path}"));
            return;
        }
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

fn find_matching_embed_end(haystack: &[u8]) -> Option<usize> {
    let open = b"[[EMBED:";
    let close = b"[[/EMBED]]";
    let mut i = 0usize;
    let mut depth = 0usize;
    while i < haystack.len() {
        if let Some(pos) = find_bytes(&haystack[i..], open) {
            let p = i + pos;
            if let Some(cpos) = find_bytes(&haystack[i..], close) {
                let c = i + cpos;
                if c < p {
                    if depth == 0 {
                        return Some(c);
                    }
                    depth -= 1;
                    i = c + close.len();
                    continue;
                }
            }
            depth += 1;
            i = p + open.len();
            continue;
        }
        if let Some(cpos) = find_bytes(&haystack[i..], close) {
            let c = i + cpos;
            if depth == 0 {
                return Some(c);
            }
            depth -= 1;
            i = c + close.len();
            continue;
        }
        break;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{EmbeddedLimits, collect_embedded};

    #[test]
    fn recursive_embedded_and_limits_work() {
        let input = br#"head[[EMBED:doc1]]a[[EMBED:doc2]]b[[/EMBED]]c[[/EMBED]]tail"#;
        let (nodes, warnings) = collect_embedded(
            input,
            &EmbeddedLimits {
                max_depth: 4,
                max_files: 10,
            },
        );
        assert_eq!(warnings.len(), 0);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].path, "root/doc1");
        assert_eq!(nodes[1].path, "root/doc1/doc2");
    }

    #[test]
    fn embedded_depth_limit_isolated() {
        let input = br#"[[EMBED:d1]][[EMBED:d2]]x[[/EMBED]][[/EMBED]]"#;
        let (_nodes, warnings) = collect_embedded(
            input,
            &EmbeddedLimits {
                max_depth: 1,
                max_files: 10,
            },
        );
        assert!(warnings.iter().any(|w| w.starts_with("embedded-depth-limit:")));
    }
}
