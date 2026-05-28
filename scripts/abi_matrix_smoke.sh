#!/usr/bin/env bash
set -euo pipefail

report="${1:-docs/dev/1-abi-matrix-report.md}"
date_str="$(date +%F)"

cargo build --release -p vectraparse-ffi
gcc examples/c/smoke.c -Iinclude -Ltarget/release -lvectraparse_ffi -Wl,-rpath,'$ORIGIN/../target/release' -o target/smoke-c
LD_LIBRARY_PATH=target/release ./target/smoke-c >/dev/null

nm -g target/release/libvectraparse_ffi.a | rg "vectraparse_(detect|parse|result_free|version)" >/dev/null
nm -D target/release/libvectraparse_ffi.so | rg "vectraparse_(detect|parse|result_free|version)" >/dev/null

cat > "${report}" <<EOF
# ABI Matrix Smoke Report

- Date: ${date_str}
- staticlib build/link: pass
- cdylib build/link: pass
- C smoke consumer: pass
- symbol export check: pass
- optional consumers (Go/Python/JNI): deferred to dedicated env matrix.
EOF

echo "abi matrix smoke completed: ${report}"
