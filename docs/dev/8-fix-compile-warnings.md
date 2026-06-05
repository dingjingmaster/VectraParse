# 编译 warning 清理修复记录

> 文档元数据
> - 文件编号：8
> - 文档类型：fix
> - 文件路径：docs/dev/8-fix-compile-warnings.md
> - 完成日期：2026-06-05
> - 需求级别：L2

## 1. 问题定义

- 问题现象：当前分支存在编译错误，导致 `cargo check --workspace` 无法继续暴露完整 warning 清单。
- 预期行为：工作区可以完成 `cargo check --workspace`，并清理本次改动引入或当前可见的编译 warning。
- 影响范围：`vectraparse-ocr` 及 `cargo check` 暴露出的相关 Rust crate。
- 不包含：测试执行、无关重构、warning 之外的行为性改造。

## 2. 证据与根因

- 复现方式：执行 `cargo check --workspace`。
- 证据等级：E2
- 关键日志/堆栈/输入：`EnhancementRoi` 初始化包含不存在的 `width`、`height`、`mean` 字段。
- 根因：局部结构体字段定义与新代码构造点未同步。
- 相关代码路径：`crates/vectraparse-ocr/src/lib.rs`

## 3. 修复方案

- 最小修复点：先修复阻塞编译的结构体构造错误，再按 `cargo check` 结果逐项清 warning。
- 代码逻辑改动：仅删除无效字段初始化、清理实际出现的 warning，不调整功能边界。
- 影响的使用场景：Rust 工作区编译检查。
- 不影响的使用场景：运行时功能、测试逻辑、外部接口。

## 4. 修复执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 修复阻塞 `cargo check` 的编译错误 | `cargo check --workspace` | 已完成 |
| 2 | 根据 `cargo check` 输出逐项清理 warning | `cargo check --workspace` | 已完成 |

## 5. 风险摘要

| 项 | 结论 |
|----|------|
| 风险矩阵 | L2 |
| 命令权限 | C1 |
| 高风险门禁 | 是（Rust/FFI/构建相关代码） |
| 破坏性操作 | 否 |
| 用户已有修改 | 是（基于现有工作区状态做最小合并） |
| 用户确认 | 有（用户要求处理编译 warning，且明确不需要代跑测试） |
| 副作用/风险 | 低，主要风险是误触用户正在修改的相邻逻辑，已按最小范围处理 |

## 6. 验证

- 验证环境：本地工作区，Linux，Rust `cargo check`
- 回归/相关验证：`cargo check --workspace`
- 结果：通过；先修复 `EnhancementRoi` 构造错误后，清理 `rec_candidate_cache_key` 的 `dead_code` warning，最终 `cargo check --workspace` 无 warning 完成。
- 未执行验证项：测试由用户自行执行
- 残余风险：未代跑测试；运行时行为仍需由用户按既有测试集确认

## 7. 修复总结

- 最终结果：已完成当前工作区 `cargo check --workspace` 可见的编译错误与 warning 清理
- 计划偏差：无
- 后续建议：无
