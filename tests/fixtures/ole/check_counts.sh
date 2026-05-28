#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_MIN=20

count_ext() {
  local dir="$1"
  local ext="$2"
  find "$dir" -maxdepth 1 -type f -iname "*.${ext}" | wc -l | tr -d ' '
}

DOC_COUNT="$(count_ext "$ROOT_DIR/doc" "doc")"
PPT_COUNT="$(count_ext "$ROOT_DIR/ppt" "ppt")"
XLS_COUNT="$(count_ext "$ROOT_DIR/xls" "xls")"

echo "DOC samples: ${DOC_COUNT}"
echo "PPT samples: ${PPT_COUNT}"
echo "XLS samples: ${XLS_COUNT}"

ok=true
if (( DOC_COUNT < TARGET_MIN )); then
  echo "FAIL: doc count < ${TARGET_MIN}"
  ok=false
fi
if (( PPT_COUNT < TARGET_MIN )); then
  echo "FAIL: ppt count < ${TARGET_MIN}"
  ok=false
fi
if (( XLS_COUNT < TARGET_MIN )); then
  echo "FAIL: xls count < ${TARGET_MIN}"
  ok=false
fi

if [[ "$ok" == "false" ]]; then
  exit 1
fi

echo "OK: all categories have at least ${TARGET_MIN} samples."
