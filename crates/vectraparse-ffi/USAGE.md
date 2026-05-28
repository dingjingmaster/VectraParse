# VectraParse Usage

## Rust API

Use core runtime helpers for JSON outputs:

```rust
use vectraparse_core::{detect_with_limits_json, parse_with_limits_json};

let input = b"hello world";
let detect_json = detect_with_limits_json(input, 8 * 1024 * 1024)?;
let parse_json = parse_with_limits_json(input, 8 * 1024 * 1024)?;
```

## C ABI

1. Include header: `include/vectraparse.h`
2. Link against `vectraparse_ffi` (`staticlib` or `cdylib`)
3. Create/destroy handle explicitly
4. Free every `VectraParseResult` with `vectraparse_result_free`

Minimal flow:

```c
VectraParseHandle* handle = NULL;
VectraParseResult out = {0};
VectraParseError rc = vectraparse_create_handle(&handle);
rc = vectraparse_detect(handle, bytes, len, NULL, &out);
vectraparse_result_free(&out);
vectraparse_destroy_handle(handle);
```

## Error Codes

- `VECTRAPARSE_OK` (`0`): success
- `VECTRAPARSE_NULL_POINTER` (`1`): null pointer input/output
- `VECTRAPARSE_INVALID_UTF8` (`2`): invalid UTF-8 in C strings
- `VECTRAPARSE_INTERNAL` (`255`): parser/detector internal error

## Resource Limits

- `VectraParseOptions.max_bytes` controls max input bytes.
- `max_bytes == 0` falls back to default `64 MiB`.
- Over limit returns `VECTRAPARSE_INTERNAL` with error info in JSON path where applicable.

## Features and Packaging

- Feature matrix: `FEATURE_MATRIX.md`
- `pkg-config`: `pkgconfig/vectraparse.pc`
- CMake config: `cmake/VectraParseConfig.cmake`
- License manifest: `LICENSES.manifest`

## Examples

- C smoke example: `examples/c/smoke.c`
