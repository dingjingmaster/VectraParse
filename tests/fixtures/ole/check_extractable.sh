#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${1:-./target/extract-static}"

if [[ ! -x "$BIN" ]]; then
  echo "FAIL: extract binary not found or not executable: $BIN"
  echo "hint: build with:"
  echo "  gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -ldl -lpthread -lm -o target/extract-static"
  exit 1
fi

total=0
failed=0

check_one() {
  local f="$1"
  local out
  out="$("$BIN" "$f" 2>&1 || true)"
  if grep -q "error:" <<<"$out"; then
    echo "FAIL(parse): $f"
    failed=$((failed + 1))
    return
  fi
  if grep -q "^Content:[[:space:]]*$" <<<"$out" || grep -q "Content:(empty)" <<<"$out"; then
    echo "WARN(empty): $f"
    failed=$((failed + 1))
    return
  fi
  echo "OK: $f"
}

while IFS= read -r -d '' file; do
  total=$((total + 1))
  check_one "$file"
done < <(find "$ROOT_DIR" -type f \( -iname "*.doc" -o -iname "*.ppt" -o -iname "*.xls" \) -print0)

echo "---"
echo "checked: $total"
echo "failed : $failed"

if (( total == 0 )); then
  echo "FAIL: no sample files found under $ROOT_DIR"
  exit 1
fi

if (( failed > 0 )); then
  exit 1
fi

echo "OK: all samples are extractable and non-empty."
