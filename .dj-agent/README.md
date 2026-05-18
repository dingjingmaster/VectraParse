# AGENTS.md

> Agent 行为指南（入口索引版）：默认只加载入口组合，再按阶段增量加载对应分片，减少上下文占用。

## 0. 使用方式（先看这里）

- 框架启动时仅加载：`AGENTS.md` + `.dj-agent/guide/00-routing.md` + `.dj-agent/guide/05-safety-gates.md`
- 进入具体新需求后：按 `.dj-agent/guide/00-routing.md` 的加载策略追加索引和相关 Summary。
- 进入 Research 阶段：追加加载 `.dj-agent/guide/10-research.md`
- 进入 Plan 阶段：追加加载 `.dj-agent/guide/20-plan.md`
- 进入 Code 阶段：追加加载 `.dj-agent/guide/30-code.md` + `.dj-agent/guide/60-validation.md`
- 进入 Summary 阶段：追加加载 `.dj-agent/guide/40-summary.md`；审视清单按 `.dj-agent/guide/00-routing.md` 的加载时机加载 `.dj-agent/guide/50-review-checklist.md`
- 出现争议、回退、需求变更：按需追加 `.dj-agent/guide/99-appendix.md`

## 1. 快速路由

- 需求分级与阶段触发：`.dj-agent/guide/00-routing.md`
- 安全门禁与破坏性操作禁令：`.dj-agent/guide/05-safety-gates.md`
- Research 规则：`.dj-agent/guide/10-research.md`
- Plan 规则：`.dj-agent/guide/20-plan.md`
- Code 规则：`.dj-agent/guide/30-code.md`
- Summary 规则：`.dj-agent/guide/40-summary.md`
- 多角色审视清单：`.dj-agent/guide/50-review-checklist.md`
- 验证命令约定：`.dj-agent/guide/60-validation.md`
- 回退/变更处理与附录：`.dj-agent/guide/99-appendix.md`

## 2. 全局约束（始终生效）

- 不要假设。不要隐藏困惑。揭示权衡；先快速探查，仍不确定或涉及取舍/风险时再提问。
- 简洁优先：用最少代码解决问题，不做推测性设计。
- 精准修改：只改与需求直接相关的内容，不顺手重构无关代码。
- 目标驱动：将任务写成可验证目标，循环直到验证通过。
- 与现有风格保持一致；仅清理由本次修改引入的孤儿代码。
- 小任务优先走轻流程；只有影响范围、接口、数据、安全、性能、稳定性、依赖、部署或架构发生变化时才升级流程。
- 闭环交付：每次完成修改后同步更新当前任务文档进度；验证通过后按提交模板提交本次相关变更。
- 安全门禁始终优先：任何分级裁剪都不能绕过破坏性操作禁令、工作区保护和高风险开发门禁。

## 3. 文档入口

- 产品设计文档：`docs/overview-product.md`
- 开发设计文档：`docs/overview-product-dev.md`
- 需求文档索引：`docs/dev/README.md`（不存在时，首次创建 L1+ 文档前用 `.dj-agent/dev-index-template.md` 初始化）
- 模板目录：`.dj-agent/`
- 详细设计片段：`.dj-agent/fragments/`

## 4. 模板选择

| 场景 | 模板 | 输出位置 |
|------|------|----------|
| L0 小改 | 不建文档 | 最终回复说明修改与验证 |
| L1 轻量总结 | `.dj-agent/summary-lite-template.md` | `docs/dev/[N]-summary-[slug].md` |
| 普通任务/功能（L2） | `.dj-agent/task-lite-template.md` | `docs/dev/[N]-task-[slug].md` |
| Bug/问题修复（L2） | `.dj-agent/fix-lite-template.md` | `docs/dev/[N]-fix-[slug].md` |
| L3/L4 完整流程（含复杂 bug 修复） | `.dj-agent/001-research-template.md` + `.dj-agent/002-plan-template.md` + `.dj-agent/003-summary-template.md` | 同一 `N` 编号的多份文档 |
| L4 专项评审 | 无固定模板，按评审结论记录 | `docs/dev/[N]-review-[slug].md` |
| 总产品事实 | `.dj-agent/overview-product-template.md` | `docs/overview-product.md` |
| 总开发事实 | `.dj-agent/overview-product-dev-template.md` | `docs/overview-product-dev.md` |
| 详细设计片段 | `.dj-agent/fragments/` | 按需复制到任务文档或总文档 |
| 代码提交说明 | `.dj-agent/commit-message-template.md` | commit message / 提交注释 |
| 任务索引初始化 | `.dj-agent/dev-index-template.md` | `docs/dev/README.md` |

总文档模板只记录长期稳定事实；单次任务细节写入 `docs/dev/[N]-*.md`，API/数据/部署/性能/并发/内核等详细设计按需使用 `.dj-agent/fragments/`。

Todo list 规则：不新增独立 todo list 模板；执行清单边界见 `.dj-agent/guide/20-plan.md`。

## 5. 总文档更新规则

- 影响长期产品行为、用户流程或功能边界时，更新 `docs/overview-product.md`。
- 影响架构、公共接口、数据模型、依赖、部署、长期通用验证方式、项目级验证入口或发布验证标准时，更新 `docs/overview-product-dev.md`。
- 总文档只沉淀长期稳定事实，不记录单次实现过程、临时排查、阶段性计划或可从任务文档追溯的细节。
- 仅实现细节变化时，不强制更新总文档；在任务 Summary 中记录即可。
- L1+ 只要新增任务、问题、调研、计划、总结或评审文档，就必须分配从 `1` 开始递增的文件编号并更新 `docs/dev/README.md` 索引。

## 6. 维护规则

- 新增规则时，优先写入对应分片，不在本文件堆叠细节。
- 本文件只保留：加载策略、路由索引、全局约束、模板选择与总文档更新规则。
- 当分片超过 500 行时继续拆分，并在对应分片顶部加导航链接。
