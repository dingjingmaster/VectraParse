# `.doc` Piece Table 主文本越界修复记录

> 文档元数据
> - 文件编号：3
> - 文档类型：fix
> - 文件路径：docs/dev/3-fix-doc-piece-table-main-text.md
> - 完成日期：2026-05-29
> - 需求级别：L2

## 1. 问题定义

- 问题现象：当前 `.doc` 解析在部分样本中会把正文后的尾部噪声一起拼进结果，用户反馈“`.doc` 解析有问题”；对比 `/data/code/office-core` 后发现现有主路径未按主文档字符数裁剪 Piece Table。
- 预期行为：`.doc` 结构化提取优先输出主文档正文，不应因为 Piece Table 记录范围覆盖正文外区域而额外拼入尾部噪声。
- 影响范围：`crates/vectraparse-mso-binary` 的 `.doc` 结构化提取主路径。
- 不包含：`.xls/.ppt` 提取协议改造、外层 `vectraparse-parsers` 输出结构变更、无关格式重构。

## 2. 证据与根因

- 复现方式：基于现有单测构造 WordDocument/CLX 样本，令 Piece Table 的 `cpEnd` 大于主文档 `ccpText`，旧实现会把正文后的额外字节一并解码。
- 证据等级：E1
- 关键日志/堆栈/输入：代码对比显示 `/data/code/office-core/src/extractors/doc.rs` 在解析 Piece Table 时会读取 `ccpText` 并对 `cpEnd` 做 `min(ccp_text)` 裁剪；当前 `vectraparse-mso-binary` 仅按 Piece Table 范围拼接。
- 根因：当前 `.doc` 结构化提取缺少“仅保留主文档正文字符区间”的边界约束，导致部分合法但包含正文外内容的 piece 被一并解码。
- 相关代码路径：`crates/vectraparse-mso-binary/src/lib.rs`

## 3. 修复方案

- 最小修复点：在现有 `.doc` 结构化提取前增加一条借鉴 `office-core` 的“主文本优先”路径，按 `ccpText` 截断 Piece Table；保留原有编码、清洗、OCR 与回退逻辑。
- 代码逻辑改动：补充主文档 `ccpText` 读取与按正文字符数裁剪的 Piece Table 解析；新增针对越界 piece 的回归测试。
- 影响的使用场景：Piece Table 覆盖正文外区域的 `.doc` 样本，正文后尾部噪声会减少。
- 不影响的使用场景：`.xls/.ppt` 提取、`.doc` 的 OCR/回退扫描、外层 parser 返回协议。

## 4. 修复执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 建立修复文档并固定问题边界、证据与计划 | 文档检查 | 完成 |
| 2 | 为 `.doc` 主文本越界场景补充回归测试并接入 `office-core` 风格正文截断解析 | `cargo test -p vectraparse-mso-binary -- --nocapture` | 完成 |
| 3 | 运行定向验证、检查诊断并更新文档与索引 | `cargo test -p vectraparse-mso-binary -- --nocapture`; `GetDiagnostics` | 完成 |

## 5. 风险摘要

| 项 | 结论 |
|----|------|
| 风险矩阵 | L2 |
| 命令权限 | C0 / C1 |
| 高风险门禁 | 是（Rust 二进制格式解析与边界处理） |
| 破坏性操作 | 否 |
| 用户已有修改 | 否 |
| 用户确认 | 无 |
| 副作用/风险 | 有（用户真实问题样本尚未提供，本次以最小边界修复和回归测试覆盖已识别问题模式） |

## 6. 验证

- 验证环境：本地仓库 `/data/code/VectraParse`，Linux。
- 回归/相关验证：
  - `cargo test -p vectraparse-mso-binary -- --nocapture`
  - `GetDiagnostics(file:///data/code/VectraParse/crates/vectraparse-mso-binary/src/lib.rs)`
- 结果：通过；`vectraparse-mso-binary` 51 个单测全部通过，新增 `.doc` 主文本截断回归测试通过；诊断仅有既有 `cSpell` 信息提示，无新的编译/语义错误。
- 未执行验证项：未拿到用户提供的真实失败 `.doc` 样本，未做真实样本端到端复现。
- 残余风险：`office-core` 路径当前以最小方式接入，仅覆盖“主文档字符数裁剪”这一已识别问题；若用户样本还包含其他 FIB/CLX 兼容差异，需要继续补样本验证。

## 7. 修复总结

- 最终结果：已将 `office-core` 风格的主文档字符数裁剪整合进当前 `.doc` 结构化提取主路径，并补充回归测试防止正文后尾部噪声再次被拼入。
- 计划偏差：无；实现保持在 `vectraparse-mso-binary` 内部最小补丁范围。
- 后续建议：如用户能提供失败 `.doc` 样本，优先追加真实样本回归；若后续仍发现 `.xls/.ppt` 的结构性问题，可按同样方式增量吸收 `/data/code/office-core` 的 extractor 逻辑。
