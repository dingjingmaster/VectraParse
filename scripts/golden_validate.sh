#!/usr/bin/env bash
set -euo pipefail

manifest="${1:-tests/golden/manifest.tsv}"

if [[ ! -f "$manifest" ]]; then
  echo "manifest not found: $manifest" >&2
  exit 1
fi

line_no=0
while IFS=$'\t' read -r id file media expected; do
  line_no=$((line_no + 1))
  [[ -z "${id}" || "${id}" =~ ^# ]] && continue

  if [[ -z "${file}" || -z "${media}" || -z "${expected}" ]]; then
    echo "invalid manifest entry at line ${line_no}" >&2
    exit 1
  fi
  if [[ ! -f "$file" ]]; then
    echo "missing fixture file: $file (line ${line_no})" >&2
    exit 1
  fi
  if [[ ! -f "$expected" ]]; then
    echo "missing expected file: $expected (line ${line_no})" >&2
    exit 1
  fi
done < "$manifest"

echo "manifest validation passed: $manifest"
