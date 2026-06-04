use std::env;
use std::path::PathBuf;

fn main() {
    let default_ort_install = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("..")
        .join("..")
        .join("build-build")
        .join("install");
    let ort_install = env::var("ORT_INSTALL_DIR")
        .unwrap_or_else(|_| default_ort_install.to_string_lossy().into_owned());

    let ort_install_path = PathBuf::from(&ort_install);
    let canon = ort_install_path
        .canonicalize()
        .unwrap_or_else(|_| ort_install_path.clone());

    let lib_dir = canon.join("lib");

    let static_lib = canon
        .join("static")
        .join("lib")
        .join("libonnxruntime_all.a");

    if static_lib.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            static_lib.parent().unwrap().display()
        );
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-lib=static=onnxruntime_all");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rerun-if-changed={}", static_lib.display());
    } else if lib_dir.join("libonnxruntime.so").exists() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=onnxruntime");
    } else {
        panic!(
            "libonnxruntime not found at `{}`.\n\
             Run `build-build/build_ort.sh` to build onnxruntime (shared or with --static).\n\
             Set ORT_INSTALL_DIR to a valid onnxruntime installation.",
            lib_dir.display()
        );
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ORT_INSTALL_DIR");
}
