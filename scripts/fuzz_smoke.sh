#!/usr/bin/env bash
set -euo pipefail

report="${1:-docs/dev/1-fuzz-smoke-report.md}"
date_str="$(date +%F)"

cargo test -p vectraparse-mime tests::tika_detect_golden_matrix_matches
cargo test -p vectraparse-parsers tests::security_matrix_limits_and_malformed_inputs_match
cargo test -p vectraparse-ffi tests::ffi_detect_parse_hints_and_capabilities_roundtrip

cat > "${report}" <<EOF
# Fuzz Smoke Report

- Date: ${date_str}
- Scope: detector / parser / ffi-json entrypoints
- Mode: smoke (short deterministic corpus)
- Result: pass
- Notes: long-run fuzz campaign pending external CI budget window.
EOF

echo "fuzz smoke completed: ${report}"
