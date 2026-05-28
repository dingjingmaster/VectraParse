#!/usr/bin/env bash
set -euo pipefail

report="${1:-docs/dev/1-bench-smoke-report.md}"
date_str="$(date +%F)"

start_all=$(date +%s)
start_det=$(date +%s)
cargo test -p vectraparse-mime tests::tika_detect_golden_matrix_matches >/dev/null
end_det=$(date +%s)

start_parse=$(date +%s)
cargo test -p vectraparse-parsers tests::extraction_golden_matrix_matches >/dev/null
end_parse=$(date +%s)

start_conc=$(date +%s)
cargo test -p vectraparse-ffi tests::ffi_detect_parse_hints_and_capabilities_roundtrip >/dev/null
end_conc=$(date +%s)
end_all=$(date +%s)

detect_sec=$((end_det - start_det))
parse_sec=$((end_parse - start_parse))
conc_sec=$((end_conc - start_conc))
total_sec=$((end_all - start_all))

cat > "${report}" <<EOF
# Bench Smoke Report

- Date: ${date_str}
- Detect throughput smoke (sec): ${detect_sec}
- Common parse latency smoke (sec): ${parse_sec}
- Concurrent/ABI smoke (sec): ${conc_sec}
- Total smoke duration (sec): ${total_sec}
- Regression threshold: +/- 20% vs previous smoke run (manual compare).
EOF

echo "bench smoke completed: ${report}"
