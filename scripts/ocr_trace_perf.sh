#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./target/extract-static}"
shift || true

if (( $# > 0 )); then
  SAMPLES=("$@")
else
  CANDIDATE_SAMPLES=(
    "${HOME}/files/b.png"
    "${HOME}/files/text-page.png"
    "${HOME}/files/large-page.png"
  )
  SAMPLES=()
  for sample in "${CANDIDATE_SAMPLES[@]}"; do
    if [ -f "${sample}" ]; then
      SAMPLES+=("${sample}")
    fi
  done
fi

if (( ${#SAMPLES[@]} == 0 )); then
  echo "No input samples found. Provide sample paths explicitly." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 required for ocr_trace_perf.sh" >&2
  exit 1
fi

if [ ! -x "${BIN}" ]; then
  echo "Executable not found: ${BIN}" >&2
  exit 1
fi

python3 - "$BIN" "${SAMPLES[@]}" <<'PY'
import json
import os
import subprocess
import sys
import statistics

bin_path = sys.argv[1]
samples = sys.argv[2:]

FIELDS = [
    "rec_primary_ms",
    "rec_alt_ms",
    "preprocess_ms",
    "preprocess_call_count",
    "preprocess_cache_hit_count",
    "preprocess_cache_miss_count",
    "variant_candidate_count",
    "det_ms",
    "page_region_ms",
    "tile_ms",
    "color_region_ms",
    "layered_region_ms",
    "visual_region_ms",
    "fallback_ms",
    "ort_intra_threads",
]


def extract_value(trace_json, path):
    timing = trace_json.get("timing", {})
    return timing.get(path, 0)


def run_case(path):
    if not os.path.exists(path):
        return None, f"missing input: {path}"
    env = os.environ.copy()
    env["VECTRAPARSE_OCR_TRACE"] = "1"
    env["VECTRAPARSE_OCR_TRACE_JSON"] = "1"
    proc = subprocess.run(
        [bin_path, path],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    trace_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("[OCR_TRACE_JSON] "):
            trace_line = line[len("[OCR_TRACE_JSON] ") :]
            break
    if trace_line is None:
        for line in proc.stderr.splitlines():
            if line.startswith("[OCR_TRACE_JSON] "):
                trace_line = line[len("[OCR_TRACE_JSON] ") :]
                break
    if trace_line is None:
        return None, "trace json not found"
    try:
        return json.loads(trace_line), None
    except json.JSONDecodeError as exc:
        return None, f"invalid trace json: {exc}"


print(
    "path,dims,"
    "rec_primary_ms,rec_alt_ms,preprocess_ms,preprocess_call_count,preprocess_cache_hits,preprocess_cache_miss,"
    "variant_candidates,det_ms,page_region_ms,tile_ms,color_region_ms,layered_region_ms,visual_region_ms,fallback_ms,ort_intra_threads"
)
rows = []
for path in samples:
    trace, err = run_case(path)
    if err:
        print(f"{path},ERROR,{err}")
        continue
    timing = trace.get("timing", {})
    line = [
        path,
        f"{trace.get('image', {}).get('width', 0)}x{trace.get('image', {}).get('height', 0)}",
    ]
    for field in FIELDS:
        val = timing.get(field)
        line.append(val if val is not None else 0)
    print(",".join(str(v) for v in line))
    rows.append((path, {field: extract_value(trace, field) for field in FIELDS}, timing))

if rows:
    totals = {}
    for field in FIELDS:
        values = [row[1].get(field, 0) for row in rows]
        if not values:
            continue
        totals[field] = {
            "avg": int(statistics.mean(values)),
            "min": min(values),
            "max": max(values),
        }
    print("")
    print("[summary]")
    for field, value in totals.items():
        print(f"{field}: avg={value['avg']} min={value['min']} max={value['max']}")
PY
