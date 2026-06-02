# OCR 速度优化 轻量任务记录

> 文档元数据
> - 文件编号：6
> - 文档类型：task
> - 文件路径：docs/dev/6-task-ocr-speed-optimization.md
> - 文档版本：v1.0.0
> - 最后更新：2026-06-02
> - 需求级别：L2
> - 关联需求：一次性完成 OCR 主链路速度优化

## 1. 目标

- 要解决的问题：
  - 当前 OCR 主链路存在较多 eager supplement、全图 fallback 和主/备识别模型重复运行，复杂截图耗时偏高。
- 成功标准：
  - trace 能输出主要阶段耗时。
  - 正常样本下减少不必要的 supplement / fallback 触发。
  - 局部识别默认不再无条件跑主/备双模型。
  - 保持现有构建、测试和 ABI 验证通过。

## 2. 背景与边界

- 包含：
  - `crates/vectraparse-ocr/src/lib.rs`
  - OCR trace / diagnostics
  - OCR 速度相关阈值与预算
- 不包含：
  - ONNX 模型文件
  - OCR 文本识别语义级策略重写
  - FFI C ABI 变更

## 3. 风险门禁

| 项 | 结论 |
|----|------|
| 风险矩阵 | L2 |
| 高风险开发门禁 | 是（OCR 主流程、性能/稳定性） |
| 破坏性操作 | 否 |
| 用户已有修改 | 待开始前检查 |
| 底层/系统风险 | 否 |
| 命令权限 | C1 |

## 4. 方案

- 为 OCR trace 增加阶段耗时与总耗时，便于后续持续调参。
- 抬高 eager supplement 触发门槛，优先让主 det 直接收敛。
- `recognize_best` 改为 primary 优先、alt 按需触发。
- `apply_quality_fallbacks` 增加 family 预算与提前停止，压缩最重的兜底链路。

## 5. 执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 建立任务文档与索引 | 文档检查 | 完成 |
| 2 | 实现 OCR 速度优化主链路 | `cargo test -p vectraparse-ocr`、定向 `cargo test` | 完成 |
| 3 | 更新任务记录并提交 | `git diff --check`、必要构建验证 | 完成 |

## 6. 实现记录

- 修改文件：
  - `crates/vectraparse-ocr/src/lib.rs`
  - `crates/vectraparse-parsers/src/lib.rs`
  - `docs/dev/README.md`
- 关键决策：
  - 为 `OcrTrace` 增加阶段耗时与主/备识别调用统计，优先补观测性，不直接改模型侧实现。
  - 主流程新增 eager color supplement 触发门槛，默认不再无条件跑颜色/分层补扫。
  - `recognize_best` 改为 primary 优先、alt 按需触发，减少每个 crop 的双模型重复推理。
  - `apply_quality_fallbacks` 改为 family budget 控制，空结果与部分结果使用不同预算。
  - 高分辨率 tile 与 visual supplement 的门槛收紧，但保留对可修复长行的补扫能力。
- 计划偏差：
  - `OcrTrace` 新字段影响到 parser 单测桩，已同步补齐。

## 7. 验证记录

| 验证项 | 命令/步骤 | 结果 | 备注 |
|--------|-----------|------|------|
| OCR 单测 | `cargo test -p vectraparse-ocr` | 通过 | 116 tests passed |
| Parser 定向回归 | `cargo test -p vectraparse-parsers ocr_success_metadata_records_fallback_and_low_confidence -- --nocapture` | 通过 | OCR metadata 回归通过 |
| FFI 构建检查 | `cargo check -p vectraparse-ffi` | 通过 | 下游 crate 编译通过 |
| 工作区检查 | `git diff --check` | 通过 | 无空白错误 |

## 8. 总结

- 最终结果：
  - OCR trace 已可输出总耗时、det/page/tile/color/layered/visual/fallback 阶段耗时，以及 primary/alt 识别调用次数与耗时。
  - 颜色/分层/视觉补扫从默认 eager 改为按质量信号触发，复杂截图之外的样本会少跑一批补链路。
  - 单个 crop 的主/备模型识别改为按需触发，减少重复 ONNX 推理。
  - 质量 fallback 改为 family budget，避免全图 enhancement/upscale/rotation 长链路无上限展开。
- 残余风险：
  - 当前只验证了功能回归，没有补真实样本的 OCR 耗时基准对比；速度收益需要你用现网截图再确认。
  - alt 按需触发会略微改变中英文混排边界样本的模型选择频率，后续应结合 trace 再做门槛微调。
- 后续建议：
  - 下一轮直接基于 trace 的 `timing_ms` 统计私有样本，按阶段占比继续砍最重路径。
