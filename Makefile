.PHONY: check test abi-smoke golden fuzz-smoke bench-smoke pipeline

check:
	cargo check --workspace

test:
	cargo test --workspace

file-content:
	gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -ldl -lpthread -lm -o target/extract-static

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
	bash scripts/bench_smoke.sh docs/dev/1-bench-smoke-report.md

pipeline: check test abi-smoke golden fuzz-smoke bench-smoke
