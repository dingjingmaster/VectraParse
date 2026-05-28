#!/usr/bin/env bash
set -euo pipefail

report="${1:-docs/dev/1-release-checklist-report.md}"
date_str="$(date +%F)"

cargo check --workspace
cargo test --workspace
cargo build --release -p vectraparse-ffi

test -f crates/vectraparse-ffi/LICENSES.manifest
test -f crates/vectraparse-ffi/include/vectraparse.h
test -f crates/vectraparse-ffi/pkgconfig/vectraparse.pc
test -f crates/vectraparse-ffi/cmake/VectraParseConfig.cmake

lib_size=$(stat -c%s target/release/libvectraparse_ffi.so 2>/dev/null || echo 0)
static_size=$(stat -c%s target/release/libvectraparse_ffi.a 2>/dev/null || echo 0)

cat > "${report}" <<EOF
# Release Checklist Report

- Date: ${date_str}
- Workspace check/test: pass
- Release build (ffi): pass
- Packaging assets (header/pkg-config/cmake/license): pass
- cdylib size (bytes): ${lib_size}
- staticlib size (bytes): ${static_size}
- Status: pass
EOF

echo "release checklist completed: ${report}"
