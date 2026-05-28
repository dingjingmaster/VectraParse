#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC_DIR="$ROOT_DIR/doc"
PPT_DIR="$ROOT_DIR/ppt"
XLS_DIR="$ROOT_DIR/xls"
COUNT="${1:-20}"

mkdir -p "$DOC_DIR" "$PPT_DIR" "$XLS_DIR"

gen_ole_like() {
  local out="$1"
  local marker="$2"
  local title="$3"
  # Synthetic OLE-like blob: OLE magic + marker + utf-8 text payload.
  {
    printf '\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'
    printf '%s' "$marker"
    printf '\x00\x00'
    printf '%s' "$title"
    printf '\n'
  } >"$out"
}

for i in $(seq 1 "$COUNT"); do
  gen_ole_like "$DOC_DIR/synth-$i.doc" "WordDocument" "Synthetic DOC sample #$i"
  gen_ole_like "$PPT_DIR/synth-$i.ppt" "PowerPoint Document" "Synthetic PPT sample #$i"
  gen_ole_like "$XLS_DIR/synth-$i.xls" "Workbook" "Synthetic XLS sample #$i"
done

echo "Generated $COUNT samples for each category under $ROOT_DIR"
