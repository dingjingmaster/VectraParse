# VectraParse

Rust rewrite project for Tika-like content extraction and file type detection.

## Workspace

- `crates/vectraparse-core`
- `crates/vectraparse-mime`
- `crates/vectraparse-parsers`
- `crates/vectraparse-enhance`
- `crates/vectraparse-ffi`

## Validation Entry Points

- `make check`: workspace compile checks
- `make test`: workspace tests
- `make abi-smoke`: build `cdylib/staticlib` and run C integration smoke
- `make golden`: validate golden manifest and compare minimal sample output
- `make fuzz-smoke`: placeholder target (implemented in P9-04)
- `make bench-smoke`: placeholder target (implemented in P9-05)
- `make pipeline`: run all of the above in sequence

## API and ABI Docs

- FFI usage guide: `crates/vectraparse-ffi/USAGE.md`
- C header: `crates/vectraparse-ffi/include/vectraparse.h`
- `pkg-config`: `crates/vectraparse-ffi/pkgconfig/vectraparse.pc`
- CMake: `crates/vectraparse-ffi/cmake/VectraParseConfig.cmake`
