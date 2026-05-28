# OLE 二进制 Office（DOC/PPT/XLS）提取实现计划

> 文档元数据
> - 文件编号：2
> - 文档类型：plan
> - 文件路径：docs/dev/2-plan-ole-doc-ppt-xls-extraction.md
> - 文档版本：v1.0.0
> - 最后更新：2026-05-28
> - 关联调研：`docs/dev/2-research-ole-doc-ppt-xls-extraction.md`
> - 目标约束：不引入新的系统命令/二进制依赖（`soffice`/`catdoc`/`antiword` 等）。

## 1. 目标与完成定义

### 1.1 总目标

- 在纯 Rust 链路下实现 `.doc/.ppt/.xls` 文本提取；
- 接入现有 `vectraparse-parsers` 与 FFI；
- `./target/extract-static <file>` 对三类文件输出可读正文而非骨架字段。

### 1.2 完成定义（DoD）

- `.doc`：主文档正文可读、中文不乱码、非仅 OLE 流名。
- `.ppt`：按幻灯片聚合标题/正文文本。
- `.xls`：按 sheet 输出可见单元格文本（含 SST/LABELSST 路径）。
- 回归样本集通过，且不新增系统外部依赖。

## 2. 里程碑与提交节奏

- M0：框架落地（crate 与接线）-> 提交
- M1：DOC MVP -> 提交
- M2：XLS MVP -> 提交
- M3：PPT MVP -> 提交
- M4：编码/清洗/稳定性增强 -> 提交
- M5：回归与文档收口 -> 提交

每个里程碑必须：
1) 测试通过；
2) 更新本计划 TODO 状态；
3) 提交代码。

## 3. 详细 TODO（可执行清单）

## 3.1 M0 - 基础骨架

- [x] 新建 `crates/vectraparse-mso-binary`（纯 Rust）。
- [ ] 定义模块：
  - [x] `text`（编码与清洗）
  - [ ] `ole`（CFB 读取）
  - [ ] `doc`（WW8）
  - [ ] `xls`（BIFF）
  - [ ] `ppt`（PPT Binary）
- [ ] 定义统一输出结构（建议）：
  - [x] `ExtractedText { file_type, blocks, warnings }`（以 `LegacyMsoExtract { kind, text, warnings }` 轻量实现）
  - [ ] `TextBlock { section, text, source_offset }`
- [x] 在 `vectraparse-parsers` 中预留新解析入口（当前已接入 `OleLegacyParser` 主路径）。
- [x] 增加最小单测（`vectraparse-mso-binary` 已补充基础识别/乱码修复单测）。

验收：
- crate 编译通过；
- 解析入口可被调用；
- 不影响现有 docx/png/zip 行为。

---

## 3.2 M1 - DOC MVP（优先）

- [ ] OLE 层：读取 `WordDocument`、`0Table`/`1Table` 流。
- [ ] FIB 解析：
  - [ ] 校验版本与关键 flag；
  - [ ] 读取 `fcClx/lcbClx`。
- [ ] CLX / Piece Table：
  - [ ] 解析 `Pcdt`；
  - [ ] 遍历 piece 并按字符区间拼接。
- [ ] 编码处理：
  - [ ] Unicode piece（UTF-16LE）；
  - [ ] ANSI piece（按 codepage 解码，默认 cp1252，含可配置回退）。
- [ ] 文本清洗：
  - [ ] 去除控制字符噪声；
  - [ ] 规范换行。
- [x] 在 `vectraparse-parsers` 替换 `.doc` 旧字符串扫描路径（已切换到 `vectraparse-mso-binary`）。
- [ ] 增加样例测试（中英混合 doc）。

验收：
- `test.doc` 输出真实段落文本；
- 不再出现以 `Root Entry` 等骨架字段为主的结果。

---

## 3.3 M2 - XLS MVP

- [ ] OLE 层：读取 `Workbook`（回退 `Book`）。
- [ ] BIFF record 基础框架：
  - [ ] BOF/EOF；
  - [ ] BOUNDSHEET；
  - [ ] SST + CONTINUE；
  - [ ] LABELSST/LABEL/NUMBER/RK/FORMULA(缓存值)。
- [ ] sheet 输出组织：
  - [ ] 按 sheet 名分块；
  - [ ] 按行列顺序拼接文本。
- [ ] 数值格式基础转换（避免科学计数法误读）。
- [x] 在 `vectraparse-parsers` 替换 `.xls` 旧路径（当前为 OLE 分策略提取 + 去噪）。
- [ ] 增加样例测试（含 SST 与数字单元格）。

验收：
- `.xls` 能稳定输出 sheet 文本；
- 中文内容无系统性乱码。

---

## 3.4 M3 - PPT MVP

- [ ] OLE 层：读取 `PowerPoint Document`（可选 `Current User`）。
- [ ] record 遍历器：
  - [ ] 支持 container 递归；
  - [ ] 深度与长度防护（防损坏文件）。
- [ ] 文本 atom 提取：
  - [ ] `TextBytesAtom`；
  - [ ] `TextCharsAtom`；
  - [ ] `CString`。
- [ ] 文本聚合：
  - [ ] 按 slide 顺序；
  - [ ] 段落分隔与去重。
- [x] 在 `vectraparse-parsers` 替换 `.ppt` 旧路径（当前为 OLE 分策略提取 + 去噪）。
- [ ] 增加样例测试（多页 ppt）。

验收：
- 输出可读幻灯片文字；
- 不仅是 record 名或元数据碎片。

---

## 3.5 M4 - 编码与稳定性增强

- [ ] 统一编码策略（优先 BOM/结构字段，其次统计回退）。
- [ ] 增加乱码检测与自动二次解码（仅在高置信触发）。
- [ ] 文本质量评分（可选）：当内容疑似噪声时 fallback 到次优链路。
- [ ] 错误分级：
  - [ ] `Unsupported`
  - [ ] `Corrupted`
  - [ ] `PartialExtracted`
- [ ] 增加超大文件保护（内存上限、片段输出）。

验收：
- 错误可诊断；
- 乱码比例显著下降；
- 大文件不崩溃。

---

## 3.6 M5 - 回归与交付收口

- [ ] 建立三格式回归样本目录（每类 >= 20）。
- [ ] 补全单测/集成测试：
  - [ ] 成功提取断言；
  - [ ] 空内容断言；
  - [ ] 损坏文件断言。
- [ ] `extract_static.c` 示例输出优化：
  - [ ] 打印分块（sheet/slide/section）；
  - [ ] 保留现有 file type 展示；
  - [ ] 内容为空时给出原因说明。
- [ ] 更新文档：
  - [ ] 本计划 TODO 全部打勾；
  - [ ] 新增 summary 文档（同编号 2）。

验收：
- 三格式主样本可读；
- CI/本地验证命令全部通过；
- 文档与代码状态一致。

## 4. 验证命令清单

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p vectraparse-ffi
gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -ldl -lpthread -lm -o target/extract-static
./target/extract-static <sample.doc>
./target/extract-static <sample.ppt>
./target/extract-static <sample.xls>
```

## 5. 风险与应对

- DOC piece table 偏移错误风险：
  - 对策：先做严格边界检查 + 失败时返回结构化错误，不默默扫字符串。
- XLS BIFF 版本差异风险：
  - 对策：MVP 聚焦 BIFF8，其他版本明确 `Unsupported`。
- PPT 递归 record 复杂风险：
  - 对策：统一 record walker，限制递归深度与单记录最大长度。
- 编码误判风险：
  - 对策：结构字段优先，统计回退仅作兜底并保留 warning。

## 6. 提交信息模板（每里程碑一次）

- `feat(mso): scaffold pure-rust ole binary parser crate`
- `feat(doc): implement ww8 fib+clx piece-table text extraction`
- `feat(xls): implement biff workbook text extraction`
- `feat(ppt): implement ppt record text atom extraction`
- `feat(mso): improve encoding fallback and robustness`
- `test(mso): add regression suite and finalize extraction output`
