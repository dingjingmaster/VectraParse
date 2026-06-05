# OCR 阶段耗时优化 轻量任务记录

> 文档元数据
> - 文件编号：9
> - 文档类型：task
> - 文件路径：docs/dev/9-task-ocr-stage-latency-optimization.md
> - 文档版本：v1.0.0
> - 最后更新：2026-06-05
> - 需求级别：L2
> - 关联需求：一次性落实 OCR 阶段耗时分析后的 5 个优化方向

## 1. 目标

- 要解决的问题：
  - 复杂截图样本中 `detected-text`、`page-regions`、`uncovered-color-region-detections` 阶段耗时过高，总耗时主要被重复 det pass 和大框 split 后逐框 rec 吞掉。
- 成功标准：
  - 更激进地收紧 `page-regions` 与 `color-region-det` 触发条件。
  - 大框优先 direct rec，只有 direct 不够好时才继续 split。
  - 超宽框的 split 工作量被显式限流。
  - ORT 默认线程策略更保守，减少小 crop 顺序推理场景的过度并行。
  - `cargo check --workspace` 通过。

## 2. 背景与边界

- 背景：
  - 用户提供的真实运行日志显示：`detected-text` 约 91.9s、`page-regions` 约 69.0s、`uncovered-color-region-detections` 约 80.4s，为主要热点。
- 包含：
  - `crates/vectraparse-ocr/src/lib.rs`
  - `crates/vectraparse-ocr/src/ort.rs`
  - 本次任务文档与 `docs/dev/README.md`
- 不包含：
  - ONNX 模型、字典、FFI ABI 变更
  - OCR 文本语义规则重写
  - 测试执行与基准跑数
- 关键假设：
  - 当前真实样本的主要热点是重复 det pass 和大框 split，而不是单次模型推理本身。
- 非目标：
  - 不在本轮引入新的缓存层、统一 supplement candidate pool 或新的 trace 字段。
- 最大修改范围：
  - OCR 单模块时延 heuristics 与 ORT 默认线程策略。
- 禁止触碰范围：
  - 不改 `unsafe` FFI 调用接口，不改构建/链接路径。

## 3. 风险门禁

| 项 | 结论 |
|----|------|
| 风险矩阵 | L2 |
| 高风险开发门禁 | 是（Rust OCR 主流程、性能/稳定性） |
| 破坏性操作 | 否 |
| 用户已有修改 | 否（开始时工作区状态未见额外未提交改动） |
| 底层/系统风险 | 否（未改 `unsafe`/ABI） |
| 命令权限 | C1 |
| 用户确认事项 | 有（用户明确要求一次性按 5 个方向落优化） |
| 回滚/止损方式 | 回退本次 OCR heuristics 与 ORT 默认线程策略改动即可 |

## 4. 方案

- 推荐方案：
  - 新增更严格的 `page-regions` 触发函数，已有足够文本时直接跳过整页补扫。
  - 为 `color-region-det` 增加更严格的后续触发门槛、超大 ROI worth-det 过滤和无增益提前停止。
  - 对大框/超宽框启用 direct-first，先尝试直接识别，只有 direct 结果不足时再继续 split。
  - 对超宽框收紧 per-box split budget，并禁用局部 det upscale。
  - 把 ORT 默认 intra-threads 从“约半核”改成“约四分之一核，上限 4”。
- 取舍理由：
  - 这些改动都在现有 heuristics 上做收缩，不引入新依赖、不扩展外部接口，属于最小风险的首轮降耗。
- 风险与应对：
  - 风险是补扫减少后少量难例文本可能回退；本轮先保留 direct 结果兜底，并且不改模型与 ABI，后续由你跑真实样本回归确认。

## 5. 执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 落地 `page-regions` / `color-region-det` 触发收紧 | `cargo check --workspace` | 完成 |
| 2 | 落地大框 direct-first、超宽框限流与 ORT 默认线程调整 | `cargo check --workspace` | 完成 |
| 3 | 更新任务文档与索引 | 文档检查 | 完成 |

## 6. 实现记录

- 修改文件：
  - `crates/vectraparse-ocr/src/lib.rs`
  - `crates/vectraparse-ocr/src/ort.rs`
  - `docs/dev/9-task-ocr-stage-latency-optimization.md`
  - `docs/dev/README.md`
- 关键决策：
  - `page-regions` 不再只依赖高成本补扫总开关，已有足够长文本/较多行时直接跳过。
  - `color-region-det` 在已有中等以上结果时不再默认继续，并对超大 ROI 做 worth-det 过滤。
  - `push_recognized_box_lines` 对大框改为 direct-first；direct 已足够好时不再进入高成本 split。
  - 超宽框单独限流 split budget，并禁止局部 det upscale。
  - ORT 默认线程数改为更保守的 quarter-core capped-at-4 策略。
  - 基于真实样本回归分析，单独回调 `color-region-det`：放宽 followup 触发门槛，并降低超宽/大背景 ROI 的 worth-det 过滤强度，优先恢复长背景框内文本的局部二次 det 机会。
- 计划偏差：
  - 无。
- 安全门禁执行结果：
  - 仅修改本次需求直接相关文件；未触碰 `unsafe` FFI 接口、模型文件或构建链路。

## 7. 验证记录

- 验证环境：
  - 本地工作区
- 系统信息（OS/内核/架构/编译器/运行时，按需）：
  - Linux，Rust `cargo check`

| 验证项 | 命令/步骤 | 结果 | 备注 |
|--------|-----------|------|------|
| 工作区编译检查 | `cargo check --workspace` | 通过 | 未运行测试，按用户协作方式保留给用户执行 |

- 未执行验证项：
  - 真实样本耗时回归
  - 单元测试 / 集成测试
- 残余风险：
  - 触发门槛收紧后，少量依赖 page-region 或 color-region-det 补扫的边缘样本可能回退，需要用真实截图验证。

## 8. 总结

- 最终结果：
  - 已一次性落实 5 个阶段耗时优化方向，并保持工作区编译通过。
- 遗留风险：
  - 还没有真实样本耗时和文本质量回归数据。
- 后续建议：
  - 你跑完真实样本后，如果热点仍集中在 `detected-text`，下一轮优先补 det/rec 次数级 trace 和 crop 级缓存统计。
