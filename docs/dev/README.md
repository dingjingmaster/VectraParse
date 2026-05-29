# 需求与任务索引

> 记录 L1+ 新增任务、问题、调研、计划、总结或评审文档。新需求/问题优先读取相关 Summary 或轻量任务记录，避免加载完整历史上下文。

## 编号规则

- 文件编号使用从 `1` 开始递增的正整数，不要求固定位数。
- 编号全局只在 `docs/dev/` 下递增，不按类型分别编号。
- 新增任务、问题修复、调研、计划、总结、评审或需求变更文档前，先检查本索引和 `docs/dev/` 现有文件名，取最大编号 + 1。
- 同一需求的多份文档使用同一编号，例如 `2-research-xxx.md`、`2-plan-xxx.md`、`2-summary-xxx.md`。
- 编号一旦分配不得复用；取消、废弃、拆分、合并也要在索引中保留记录并标注状态。
- 文件命名格式：`N-[type]-[slug].md`，其中 `type` 可取 `summary`、`task`、`fix`、`research`、`plan`、`review`。

## 索引

| ID | 日期 | 级别 | 类型 | 文档 | 状态 | 摘要 |
|----|------|------|------|------|------|------|
| 1 | 2026-05-27 | L4 | research | [1-research-rust-tika-rewrite.md](1-research-rust-tika-rewrite.md) | 已完成 | 调研 `/data/source/tika` 项目结构，明确 Rust 静态库/动态库重写的完整功能范围、架构边界、对照遗漏、风险和 Plan 阶段排序依据。 |
| 1 | 2026-05-27 | L4 | plan | [1-plan-rust-tika-rewrite.md](1-plan-rust-tika-rewrite.md) | 已完成 | 将 Tika Rust 重写范围整合为完整执行计划，列出基础架构、检测、解析、增强、FFI、验证和发布 todo。 |
| 2 | 2026-05-28 | L3 | research | [2-research-ole-doc-ppt-xls-extraction.md](2-research-ole-doc-ppt-xls-extraction.md) | 已完成 | 基于 LibreOffice 源码路径调研 OLE 二进制 Office（`.doc/.ppt/.xls`）可行提取方案，结论为可通过纯 Rust 重写解析链路达成且不引入新的系统二进制依赖。 |
| 2 | 2026-05-28 | L3 | plan | [2-plan-ole-doc-ppt-xls-extraction.md](2-plan-ole-doc-ppt-xls-extraction.md) | 进行中 | 基于调研拆解 `.doc/.ppt/.xls` 纯 Rust 实施 TODO，按里程碑定义验收标准、测试命令与提交节奏。 |
| 2 | 2026-05-28 | L3 | summary | [2-summary-ole-doc-ppt-xls-extraction.md](2-summary-ole-doc-ppt-xls-extraction.md) | 进行中 | 阶段性总结 DOC/XLS/PPT 结构化提取、编码与稳定性增强进展，记录当前验证范围与剩余收口事项。 |
| 3 | 2026-05-29 | L2 | fix | [3-fix-doc-piece-table-main-text.md](3-fix-doc-piece-table-main-text.md) | 已完成 | 参考 `/data/code/office-core` 的 `.doc` 提取逻辑，为 Piece Table 主文本越界场景增加 `ccpText` 裁剪，减少正文后尾部噪声并补充回归测试。 |
