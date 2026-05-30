use std::env;
use std::path::PathBuf;

fn main() {
    let ort_install =
        env::var("ORT_INSTALL_DIR").unwrap_or_else(|_| "build-build/install".to_string());

    let ort_install_path = PathBuf::from(&ort_install);
    let canon = ort_install_path
        .canonicalize()
        .unwrap_or_else(|_| ort_install_path.clone());

    let lib_dir = canon.join("lib");

    if !lib_dir.join("libonnxruntime.so").exists() {
        panic!(
            "libonnxruntime.so not found at `{}`.\n\
             Run `build-build/build_ort.sh` to build onnxruntime from source, or\n\
             extract a pre-built release to `build-build/install/`.\n\
             Set ORT_INSTALL_DIR to a valid onnxruntime installation.",
            lib_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_INSTALL_DIR");
}
