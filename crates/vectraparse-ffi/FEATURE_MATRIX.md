# Feature Matrix (vectraparse-ffi)

| Feature | Default | Effect |
|---|---:|---|
| `default` | yes | Enables core detect/parse + JSON results. |
| `staticlib` | yes (crate-type) | Build static library for C/C++ linking. |
| `cdylib` | yes (crate-type) | Build shared library for dynamic linking. |
| `rlib` | yes (crate-type) | Rust internal linking artifact. |

## Native/Service Capability Policy

- External services are disabled by default unless explicitly wired at upper layer.
- Resource limits are always enforced from `VectraParseOptions.max_bytes`.
