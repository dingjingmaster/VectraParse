#!/usr/bin/env bash
set -euo pipefail

expected="${1:-}"
actual="${2:-}"

if [[ -z "$expected" || -z "$actual" ]]; then
  echo "usage: bash scripts/golden_compare.sh <expected.json> <actual.json>" >&2
  exit 1
fi

if [[ ! -f "$expected" ]]; then
  echo "expected file not found: $expected" >&2
  exit 1
fi
if [[ ! -f "$actual" ]]; then
  echo "actual file not found: $actual" >&2
  exit 1
fi

if diff -u "$expected" "$actual"; then
  echo "golden compare passed"
else
  echo "golden compare failed" >&2
  exit 1
fi
