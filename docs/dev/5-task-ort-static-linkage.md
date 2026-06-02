# ONNX Runtime 静态链接 轻量任务记录

> 文档元数据
> - 文件编号：5
> - 文档类型：task
> - 文件路径：docs/dev/5-task-ort-static-linkage.md
> - 文档版本：v1.0.0
> - 最后更新：2026-06-02
> - 需求级别：L2
> - 关联需求：将 ONNX Runtime 从动态库消费切换为静态库消费，并归并到项目静态链接产物

## 1. 目标

- 要解决的问题：
  - 当前 `build-build/install/lib/libonnxruntime.so` 仅以动态库方式参与链接，`libvectraparse_ffi.a` 的 C 消费方仍需额外提供 `-lonnxruntime` 和运行时 `rpath`。
- 成功标准：
  - `build-build/build_ort.sh --static` 可稳定生成 `build-build/install/static/lib/libonnxruntime_all.a`。
  - 项目 release 构建可在静态 ORT 模式下完成。
  - C 示例可仅链接项目侧归并后的静态库完成构建与运行，不依赖 `libonnxruntime.so`。

## 2. 背景与边界

- 背景：
  - 仓库已有静态 ORT 探测入口，但当前安装目录仍只有 `.so`，静态归并脚本也未完成收口。
- 包含：
  - `build-build/` 静态 ORT 构建链路
  - OCR build script 的静态优先链接路径
  - 项目静态归并产物与 C smoke / example 链路
  - 相关任务文档与索引
- 不包含：
  - OCR 识别逻辑
  - ONNX 模型文件内容
  - 新增外部依赖
- 关键假设：
  - 当前 CPU-only ORT 配置生成的静态库可用于现有 OCR 调用路径。
- 非目标：
  - 产出完全静态的最终 ELF（glibc 全静态不在本次范围内）。
- 最大修改范围：
  - `build-build/`、`crates/vectraparse-ocr/build.rs`、`Makefile`、`scripts/`、必要的使用说明。
- 禁止触碰范围：
  - OCR 核心算法、FFI C ABI 定义、无关 parser 行为。

## 3. 风险门禁

| 项 | 结论 |
|----|------|
| 风险矩阵 | L2 |
| 高风险开发门禁 | 是（构建系统、链接参数、ABI 交付链路） |
| 破坏性操作 | 否 |
| 用户已有修改 | 否（开始前 `git status --short` 为空） |
| 底层/系统风险 | 是（ABI/API / 构建链接） |
| 命令权限 | C1 |
| 用户确认事项 | 无 |
| 回滚/止损方式 | 回退本次构建脚本与链接脚本改动，恢复动态库消费链路 |

## 4. 方案

- 推荐方案：
  - 保留 ORT shared fallback，但补齐 `--static` 构建产物。
  - 为项目新增归并后的单一静态包产物，将 `libvectraparse_ffi.a` 与 `libonnxruntime_all.a` 归并，供 C 消费方直接链接。
- 取舍理由：
  - 不覆盖 Cargo 原始 `staticlib` 产物，降低对 Rust 构建约定的侵入。
  - 用单独归并包解决“最终消费者仍需手写 ORT 静态库”的问题。
- 风险与应对：
  - 静态 archive 归并可能出现成员冲突：使用 `ar -M` 的 `ADDLIB` 归并，避免手工解包覆盖。
  - 静态切换后可能残留旧 `.so` 路径：验证以无 `-lonnxruntime` 的 C 构建与运行结果为准。

## 5. 执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 建立任务文档与计划，明确静态构建和归并交付目标 | 文档检查 | 完成 |
| 2 | 修复 ORT 静态构建脚本并补齐项目静态归并脚本/Makefile/ABI smoke | `bash build-build/build_ort.sh --static`、`cargo build --release -p vectraparse-ffi`、C 编译 smoke | 完成 |
| 3 | 跑完整静态构建与示例验证，更新文档、索引并提交 | `make abi-smoke`、`make file-content` 或等价命令、`git diff --check` | 完成 |

## 6. 实现记录

- 修改文件：
  - `build-build/build_ort.sh`
  - `crates/vectraparse-ocr/build.rs`
  - `scripts/bundle_static_ffi.sh`
  - `Makefile`
  - `scripts/abi_matrix_smoke.sh`
  - `crates/vectraparse-ffi/USAGE.md`
  - `tests/fixtures/ole/check_extractable.sh`
  - `README.md`
  - `3thrd/date/CMakeLists.txt`
  - `3thrd/date/include/date/date.h`
  - `docs/dev/README.md`
- 关键决策：
  - 保留 `build-build/install/lib` 下既有 shared 产物，不做清理；静态消费统一走 `build-build/install/static/lib/libonnxruntime_all.a`。
  - 用 `ar -M` 归并 `libvectraparse_ffi.a` 与 `libonnxruntime_all.a`，输出单独的 `target/release/libvectraparse_ffi_full.a`，不覆盖 Cargo 默认 `staticlib`。
  - 静态 archive 消费侧统一改用 `g++` 链接，避免 C++ 运行时符号由下游手工补齐。
  - `build_ort.sh --static` 显式构建 `re2` 静态目标，再做归并，确保 ORT 静态包闭合。
  - `bundle_static_ffi.sh` 改为临时文件生成后原子 `mv`，避免并发验证时归档截断。
- 计划偏差：
  - ORT 的 `FetchContent` 在离线仓库场景下仍会打印远端 URL，但实际通过本地 `FETCHCONTENT_SOURCE_DIR_*` 覆盖完成构建。
  - 初版归并遗漏了 `re2` 静态目标，导致 bundled archive 和 `cdylib` 消费侧存在未解析符号；已通过显式构建 `re2` 收口。
- 安全门禁执行结果：
  - 未执行破坏性操作。
  - 所有修改均限定在工作区内。

## 7. 验证记录

- 验证环境：
  - 工作目录：`/data/code/VectraParse`
- 系统信息（OS/内核/架构/编译器/运行时，按需）：
  - 架构：`x86_64`
  - C/C++ 编译器：`gcc/g++ 15.2.1`
  - 构建系统：`cmake 4.3.3`、`ninja`

| 验证项 | 命令/步骤 | 结果 | 备注 |
|--------|-----------|------|------|
| ORT 静态构建 | `bash build-build/build_ort.sh --static` | 通过 | 生成 `build-build/install/static/lib/libonnxruntime_all.a` |
| Rust FFI release 构建 | `cargo build --release -p vectraparse-ffi` | 通过 | `vectraparse-ocr` 静态链接 ORT 完成 |
| cdylib 运行时依赖检查 | `ldd target/release/libvectraparse_ffi.so` | 通过 | 无 `libonnxruntime.so` 依赖 |
| bundled archive 生成 | `bash scripts/bundle_static_ffi.sh` | 通过 | 生成 `target/release/libvectraparse_ffi_full.a` |
| ABI smoke | `make abi-smoke` | 通过 | `cdylib` 与 bundled static archive 的 C smoke 均通过 |
| ABI matrix 脚本 | `bash scripts/abi_matrix_smoke.sh /tmp/abi-matrix-report.md` | 通过 | 产出报告并完成符号检查 |
| 静态示例构建 | `make file-content` | 通过 | 生成 `target/extract-static` |
| 静态示例运行 | `./target/extract-static /data/code/VectraParse/tests/fixtures/minimal.pdf` | 通过 | 输出 `application/pdf` 与 `%PDF-1.7` |
| 工作区格式检查 | `git diff --check` | 通过 | 无空白错误 |

- 未执行验证项：
  - 未做跨平台矩阵验证（当前仅验证本机 Linux/x86_64）。
- 残余风险：
  - `libonnxruntime_all.a` 中部分汇编对象仍触发 `.note.GNU-stack` 链接告警，但不影响当前静态链接结果。
  - `build-build/install/lib` 下旧 shared 产物仍保留，若后续需要彻底移除动态路径，应单独做安装目录清理策略。

## 8. 总结

- 最终结果：
  - ONNX Runtime 已可稳定构建为静态归并库，并通过 `vectraparse-ocr` 的静态链接路径接入 release 构建。
  - 项目新增 `target/release/libvectraparse_ffi_full.a`，C 消费方可直接链接，不再依赖 `libonnxruntime.so`。
  - `make abi-smoke` 与 `make file-content` 已切换到可工作的静态消费方式。
- 遗留风险：
  - 当前 bundled archive 为 Linux/x86_64 本机构建验证结果，其他平台仍需单独验证。
- 后续建议：
  - 若需要对外发布单一静态包，可继续补 `pkg-config` / CMake 导出项，使 `libvectraparse_ffi_full.a` 成为正式交付产物。
