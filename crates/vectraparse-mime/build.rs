use std::env;
use std::fs;
use std::path::PathBuf;

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

    let out = format!(
        "pub const TIKA_MIME_XML_PATH: &str = \"{source}\";\n\
         pub const MIME_TYPE_COUNT: usize = {mime_type_count};\n\
         pub const MAGIC_COUNT: usize = {magic_count};\n\
         pub const GLOB_COUNT: usize = {glob_count};\n\
         pub const ROOT_XML_COUNT: usize = {root_xml_count};\n\
         pub const ALIAS_COUNT: usize = {alias_count};\n\
         pub const SUBCLASS_COUNT: usize = {subclass_count};\n"
    );
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_path.join("mime_generated.rs"), out).expect("write generated file");
}
