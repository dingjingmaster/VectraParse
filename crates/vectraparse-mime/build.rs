use std::env;
use std::fs;
use std::path::PathBuf;

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)?;
    let value_start = start + needle.len();
    let rest = &tag[value_start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn main() {
    let source = env::var("TIKA_MIME_XML")
        .unwrap_or_else(|_| "/data/source/tika/tika-core/src/main/resources/org/apache/tika/mime/tika-mimetypes.xml".to_string());
    println!("cargo:rerun-if-env-changed=TIKA_MIME_XML");
    println!("cargo:rerun-if-changed={source}");

    let xml = fs::read_to_string(&source)
        .unwrap_or_else(|e| panic!("failed to read tika mime xml from {}: {}", source, e));
    let mime_type_count = xml.matches("<mime-type ").count();
    let magic_count = xml.matches("<magic").count();
    let glob_count = xml.matches("<glob ").count();
    let root_xml_count = xml.matches("<root-XML ").count();
    let alias_count = xml.matches("<alias ").count();
    let subclass_count = xml.matches("<sub-class-of ").count();
    let mut alias_pairs = Vec::new();
    let mut subclass_pairs = Vec::new();
    let mut mime_types = Vec::new();

    let mut pos = 0usize;
    while let Some(rel) = xml[pos..].find("<mime-type ") {
        let start = pos + rel;
        let open_end = match xml[start..].find('>') {
            Some(i) => start + i,
            None => break,
        };
        let open_tag = &xml[start..=open_end];
        if let Some(canonical) = extract_attr(open_tag, "type") {
            let canonical_norm = canonical.to_ascii_lowercase();
            mime_types.push(canonical_norm.clone());
            if open_tag.ends_with("/>") {
                pos = open_end + 1;
                continue;
            }
            let close_tag = "</mime-type>";
            let block_end = match xml[open_end + 1..].find(close_tag) {
                Some(i) => open_end + 1 + i,
                None => break,
            };
            let body = &xml[open_end + 1..block_end];

            let mut alias_pos = 0usize;
            while let Some(alias_rel) = body[alias_pos..].find("<alias ") {
                let alias_start = alias_pos + alias_rel;
                let alias_end_rel = match body[alias_start..].find('>') {
                    Some(i) => i,
                    None => break,
                };
                let alias_tag = &body[alias_start..=alias_start + alias_end_rel];
                if let Some(alias_type) = extract_attr(alias_tag, "type") {
                    alias_pairs.push((alias_type.to_ascii_lowercase(), canonical_norm.clone()));
                }
                alias_pos = alias_start + alias_end_rel + 1;
            }

            let mut sc_pos = 0usize;
            while let Some(sc_rel) = body[sc_pos..].find("<sub-class-of ") {
                let sc_start = sc_pos + sc_rel;
                let sc_end_rel = match body[sc_start..].find('>') {
                    Some(i) => i,
                    None => break,
                };
                let sc_tag = &body[sc_start..=sc_start + sc_end_rel];
                if let Some(parent) = extract_attr(sc_tag, "type") {
                    subclass_pairs.push((canonical_norm.clone(), parent.to_ascii_lowercase()));
                }
                sc_pos = sc_start + sc_end_rel + 1;
            }
            pos = block_end + close_tag.len();
        } else {
            pos = open_end + 1;
        }
    }
    mime_types.sort();
    mime_types.dedup();
    alias_pairs.sort();
    alias_pairs.dedup();
    subclass_pairs.sort();
    subclass_pairs.dedup();

    let mime_types_array = mime_types
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let alias_array = alias_pairs
        .iter()
        .map(|(a, c)| format!("(\"{a}\", \"{c}\")"))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let subclass_array = subclass_pairs
        .iter()
        .map(|(child, parent)| format!("(\"{child}\", \"{parent}\")"))
        .collect::<Vec<_>>()
        .join(",\n    ");

    let out = format!(
        "pub const TIKA_MIME_XML_PATH: &str = \"{source}\";\n\
         pub const MIME_TYPE_COUNT: usize = {mime_type_count};\n\
         pub const MAGIC_COUNT: usize = {magic_count};\n\
         pub const GLOB_COUNT: usize = {glob_count};\n\
         pub const ROOT_XML_COUNT: usize = {root_xml_count};\n\
         pub const ALIAS_COUNT: usize = {alias_count};\n\
         pub const SUBCLASS_COUNT: usize = {subclass_count};\n\
         pub const KNOWN_MIME_TYPES: &[&str] = &[\n    {mime_types_array}\n];\n\
         pub const ALIAS_PAIRS: &[(&str, &str)] = &[\n    {alias_array}\n];\n\
         pub const SUBCLASS_PAIRS: &[(&str, &str)] = &[\n    {subclass_array}\n];\n"
    );
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_path.join("mime_generated.rs"), out).expect("write generated file");
}
