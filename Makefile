.PHONY: debug release file-content bundle-static check test abi-smoke golden fuzz-smoke bench-smoke pipeline ort-build ort-check ort-test

ORT_INSTALL_DIR := $(shell pwd)/build-build/install
export ORT_INSTALL_DIR

release:
	cargo build --release

debug:
	cargo build

check:
	cargo check --workspace

test:
	cargo test --workspace

ort-build:
	bash build-build/build_ort.sh --static

ort-check:
	cargo check --workspace

ort-test:
	cargo test --workspace

bundle-static:
	bash scripts/bundle_static_ffi.sh

file-content:
	cargo build --release -p vectraparse-ffi
	bash scripts/bundle_static_ffi.sh
	g++ examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi_full.a \
		-ldl -lpthread -lm -o target/extract-static

abi-smoke:
	cargo build --release -p vectraparse-ffi
	bash scripts/bundle_static_ffi.sh
	g++ examples/c/smoke.c -Iinclude -Ltarget/release -lvectraparse_ffi \
		-Wl,-rpath,'$$ORIGIN/../target/release' -o target/smoke-c
	./target/smoke-c
	g++ examples/c/smoke.c -Iinclude target/release/libvectraparse_ffi_full.a \
		-ldl -lpthread -lm -o target/smoke-static
	./target/smoke-static

golden:
	bash scripts/golden_validate.sh tests/golden/manifest.tsv
	./target/smoke-c | sed -n '1p' | sed 's/^detect: //' > /tmp/minimal_pdf.actual.json
	bash scripts/golden_compare.sh tests/golden/expected/minimal_pdf.detect.json /tmp/minimal_pdf.actual.json

fuzz-smoke:
	bash scripts/fuzz_smoke.sh docs/dev/1-fuzz-smoke-report.md

bench-smoke:
	bash scripts/bench_smoke.sh docs/dev/1-bench-smoke-report.md

pipeline: check test abi-smoke golden fuzz-smoke bench-smoke
