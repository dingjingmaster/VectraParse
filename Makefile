.PHONY: check test abi-smoke golden fuzz-smoke bench-smoke pipeline

check:
	cargo check --workspace

test:
	cargo test --workspace

abi-smoke:
	cargo build --release -p vectraparse-ffi
	gcc examples/c/smoke.c -Iinclude -Ltarget/release -lvectraparse_ffi -Wl,-rpath,'$$ORIGIN/../target/release' -o target/smoke-c
	LD_LIBRARY_PATH=target/release ./target/smoke-c

golden:
	bash scripts/golden_validate.sh tests/golden/manifest.tsv
	LD_LIBRARY_PATH=target/release ./target/smoke-c | sed -n '1p' | sed 's/^detect: //' > /tmp/minimal_pdf.actual.json
	bash scripts/golden_compare.sh tests/golden/expected/minimal_pdf.detect.json /tmp/minimal_pdf.actual.json

fuzz-smoke:
	bash scripts/fuzz_smoke.sh docs/dev/1-fuzz-smoke-report.md

bench-smoke:
	@echo "bench-smoke placeholder: benches will be added in P9-05"

pipeline: check test abi-smoke golden fuzz-smoke bench-smoke
