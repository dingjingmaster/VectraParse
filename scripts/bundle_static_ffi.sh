#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FFI_LIB="${1:-$PROJECT_ROOT/target/release/libvectraparse_ffi.a}"
ORT_LIB="${2:-$PROJECT_ROOT/build-build/install/static/lib/libonnxruntime_all.a}"
OUT_LIB="${3:-$PROJECT_ROOT/target/release/libvectraparse_ffi_full.a}"

if [ ! -f "$FFI_LIB" ]; then
    echo "missing ffi archive: $FFI_LIB" >&2
    exit 1
fi

if [ ! -f "$ORT_LIB" ]; then
    echo "missing onnxruntime archive: $ORT_LIB" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_LIB")"
tmp_out="$(mktemp "$(dirname "$OUT_LIB")/.libvectraparse_ffi_full.a.XXXXXX")"

tmp_mri="$(mktemp)"
cleanup() {
    rm -f "$tmp_mri"
    rm -f "$tmp_out"
}
trap cleanup EXIT

{
    printf 'CREATE %s\n' "$tmp_out"
    printf 'ADDLIB %s\n' "$FFI_LIB"
    printf 'ADDLIB %s\n' "$ORT_LIB"
    printf 'SAVE\n'
    printf 'END\n'
} > "$tmp_mri"

ar -M < "$tmp_mri"
ranlib "$tmp_out"
mv "$tmp_out" "$OUT_LIB"

echo "bundled static archive: $OUT_LIB"
