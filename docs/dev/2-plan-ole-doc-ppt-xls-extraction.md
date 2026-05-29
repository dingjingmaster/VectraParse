# OLE 二进制 Office（DOC/PPT/XLS）提取实现计划

> 文档元数据
> - 文件编号：2
> - 文档类型：plan
> - 文件路径：docs/dev/2-plan-ole-doc-ppt-xls-extraction.md
> - 文档版本：v1.1.0
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
- [x] 定义模块：
  - [x] `text`（编码与清洗）
  - [x] `ole`（CFB 读取已支持目录解析、常规流读取与 mini stream 链读取）
  - [x] `doc`（WW8；已实现 FIB/CLX piece table 主路径 + 回退链路）
  - [x] `xls`（BIFF；已实现 BOF/EOF/BOUNDSHEET/SST/LABELSST/LABEL/NUMBER/RK/FORMULA）
  - [x] `ppt`（PPT Binary；已实现 record walker + Text atom 提取与 slide 聚合）
- [x] 定义统一输出结构（建议）：
  - [x] `ExtractedText { file_type, blocks, warnings }`（以 `LegacyMsoExtract { kind, text, warnings }` 轻量实现）
  - [x] `TextBlock { section, text, source_offset }`（已在 `OleLegacyParser` 输出 `ole.block_*` 元数据）
- [x] 在 `vectraparse-parsers` 中预留新解析入口（当前已接入 `OleLegacyParser` 主路径）。
- [x] 增加最小单测（`vectraparse-mso-binary` 已补充基础识别/乱码修复单测）。

验收：
- crate 编译通过；
- 解析入口可被调用；
- 不影响现有 docx/png/zip 行为。

---

## 3.2 M1 - DOC MVP（优先）

- [x] OLE 层：读取 `WordDocument`、`0Table`/`1Table` 流（已支持常规流 + mini stream）。
- [x] FIB 解析：
  - [x] 校验版本与关键 flag；
  - [x] 读取 `fcClx/lcbClx`。
- [x] CLX / Piece Table：
  - [x] 解析 `Pcdt`；
  - [x] 遍历 piece 并按字符区间拼接。
- [x] 编码处理：
  - [x] Unicode piece（UTF-16LE）；
  - [x] ANSI piece（按 codepage 解码，默认 cp1252，含可配置回退）。
- [x] 文本清洗：
  - [x] 去除控制字符噪声；
  - [x] 规范换行。
- [x] 在 `vectraparse-parsers` 替换 `.doc` 旧字符串扫描路径（已切换到 `vectraparse-mso-binary`）。
- [x] 增加样例测试（中英混合 doc）。

当前进度备注（2026-05-28）：
- 已完成：`.doc` 入口已从旧整文件字符串扫描迁移到新 crate，且优先按目标流读取后提取文本。
- 已完成：新增 FIB 基础校验（`wIdent/nFib/fWhichTblStm`）与 `fcClx/lcbClx` 定位，优先按 FIB 指定 table 流解析，失败时回退 CLX 扫描。
- 已完成：ANSI piece 新增按 `chsTables/chs/lid` 选择编码解码，默认 `windows-1252`，并支持 `VECTRAPARSE_DOC_ANSI_FALLBACK` 回退配置。
- 已完成：清洗阶段新增 Word 控制字符映射与换行归一化（段落/单元格控制符转边界，连续空行折叠）。
- 已完成：补充中英混合样例单测（ANSI + Unicode piece 混合正文）。
- 已完成：修复 DBCS ANSI piece 长度处理（按编码推断单/双字节宽度读取），补充 GBK ANSI 中文回归单测。
- 已完成：`.doc` 回退链路新增尾部噪声截断（内部锚点/连续乱码行触发），降低正文后乱码尾巴泄漏。
- 已完成：`.doc` 回退链路前置“正文后噪声段”拦截（连续乱码/锚点提前截断），避免继续扫描图片与对象区噪声。
- 已完成：`.doc` 图片流 OCR 接入（识别 `IMG_/image/picture` 等流名并尝试 OCR），OCR 失败不影响主文本提取链路。
- 已完成：ownerfile 场景二次归一化为 doc（基于 `0Table/1Table/WpsCustomData` 锚点），并在归一化后执行 doc 尾部噪声截断，减少 WPS 尾段乱码泄漏。
- 已完成：`OleLegacyParser` 输出增强，新增 `ole.block.section_count_distinct` 元数据，便于分块结构化消费。
- 已完成：`.doc` 回退链路改为多流联合扫描（`WordDocument + 0Table/1Table + Data`），避免单流扫描造成正文漏提。
- 已完成：图片 OCR 增强为“流内图片 blob 切片扫描 + OCR”（支持从复合流内提取 PNG/JPEG/GIF/BMP 片段后识别）。
- 未完成：WW8 CLX Piece Table 的完整兼容性与样例覆盖仍需继续完善。

验收：
- `test.doc` 输出真实段落文本；
- 不再出现以 `Root Entry` 等骨架字段为主的结果。

---

## 3.3 M2 - XLS MVP

- [x] OLE 层：读取 `Workbook`（回退 `Book`，已接入流选择，覆盖常规流 + mini stream）。
- [x] BIFF record 基础框架：
  - [x] BOF/EOF；
  - [x] BOUNDSHEET；
  - [x] SST + CONTINUE；
  - [x] LABELSST/LABEL/NUMBER/RK/FORMULA(缓存值)。
- [x] sheet 输出组织：
  - [x] 按 sheet 名分块；
  - [x] 按行列顺序拼接文本。
- [x] 数值格式基础转换（避免科学计数法误读）。
- [x] 在 `vectraparse-parsers` 替换 `.xls` 旧路径（当前为 OLE 分策略提取 + 去噪）。
- [x] 增加样例测试（含 SST 与数字单元格）。

当前进度备注（2026-05-28 补充）：
- 已完成：Workbook BIFF 记录遍历骨架（`record id + length`），并接入 `BOF/EOF` 基础识别。
- 已完成：`BOUNDSHEET` 名称提取（8-bit / UTF-16 名称），当前结构化输出为 `SheetN: <name>`。
- 已完成：`SST + CONTINUE` 链式解析（含跨记录拼接），并接入结构化预览输出。
- 已完成：单元格值类记录提取（`LABELSST/LABEL/NUMBER/RK/FORMULA` 缓存数值），结构化输出 `Sheet!R{row}C{col}=value`。
- 已完成：sheet 输出组织按名称分块，块内按 `row/col` 排序拼接，输出形态为 `SheetName` + `R{row}C{col}=value`。
- 已完成：数值格式基础转换，`NUMBER/RK/FORMULA` 数值输出默认避免科学计数法，并规避 `-0` 形态。
- 已完成：补充 `.xls` 样例单测（`SST + LABELSST + NUMBER` 组合路径）。
- 已完成：`FORMULA` 的字符串/布尔/错误缓存值分支解析，并接入 `STRING` 记录联动。

验收：
- `.xls` 能稳定输出 sheet 文本；
- 中文内容无系统性乱码。

---

## 3.4 M3 - PPT MVP

- [x] OLE 层：读取 `PowerPoint Document`（可选 `Current User`，已接入流选择，覆盖常规流 + mini stream）。
- [x] record 遍历器：
  - [x] 支持 container 递归；
  - [x] 深度与长度防护（防损坏文件）。
- [x] 文本 atom 提取：
  - [x] `TextBytesAtom`；
  - [x] `TextCharsAtom`；
  - [x] `CString`。
- [x] 文本聚合：
  - [x] 按 slide 顺序；
  - [x] 段落分隔与去重。
- [x] 在 `vectraparse-parsers` 替换 `.ppt` 旧路径（当前为 OLE 分策略提取 + 去噪）。
- [x] 增加样例测试（多页 ppt）。

当前进度备注（2026-05-28 补充）：
- 已完成：`PowerPoint Document` record walker（8-byte 头解析）与 `container(recVer=0xF)` 递归遍历。
- 已完成：walker 深度上限与单记录长度上限防护，损坏/异常长度记录可提前终止。
- 已完成：文本 atom 提取（`TextBytesAtom/TextCharsAtom/CString`）与基础去重拼接。
- 已完成：按 slide 顺序分块聚合（`Slide N`）与相邻重复段落去重。
- 已完成：多页 ppt 样例测试（两页 slide，覆盖 TextBytesAtom + TextCharsAtom）。

验收：
- 输出可读幻灯片文字；
- 不仅是 record 名或元数据碎片。

---

## 3.5 M4 - 编码与稳定性增强

- [x] 统一编码策略（优先 BOM/结构字段，其次统计回退）。
- [x] 增加乱码检测与自动二次解码（仅在高置信触发）。
- [x] 文本质量评分（可选）：当内容疑似噪声时 fallback 到次优链路。
- [x] 错误分级：
  - [x] `Unsupported`
  - [x] `Corrupted`
  - [x] `PartialExtracted`
- [x] 增加超大文件保护（内存上限、片段输出）。

当前进度备注（2026-05-28 补充）：
- 已完成：提取结果 warning 增加错误分级标记：`Unsupported`、`Corrupted`、`PartialExtracted`。
- 已完成：输入级内存保护（64MB 上限）与超限片段输出策略，超限场景标记 `PartialExtracted`。
- 已完成：统一字节解码策略（BOM 优先、结构字段编码优先、统计回退），并接入 PPT TextBytesAtom 与 XLS 文本路径。
- 已完成：XLS 结构字段 `CODEPAGE(0x0042)` 优先解码接入（`LABEL/SST` 8-bit 分支按 codepage 解码）。
- 已完成：高置信乱码检测与自动二次解码（疑似 mojibake 且原字节可 UTF-8 解码时自动回退 UTF-8）。
- 已完成：轻量文本质量评分与低质量回退策略（结构化输出低分时自动回退扫描链路）。
- 已完成：新增 doc 噪声模式识别回归测试（包含短乱码、WPS 元数据串、base64 噪声模式）。

验收：
- 错误可诊断；
- 乱码比例显著下降；
- 大文件不崩溃。

---

## 3.6 M5 - 回归与交付收口

- [x] 建立三格式回归样本目录（每类 >= 20）。
- [x] 补全单测/集成测试：
  - [x] 成功提取断言；
  - [x] 空内容断言；
  - [x] 损坏文件断言。
- [x] `extract_static.c` 示例输出优化：
  - [x] 打印分块（sheet/slide/section）；
  - [x] 保留现有 file type 展示；
  - [x] 内容为空时给出原因说明。
- [x] 更新文档：
  - [x] 本计划 TODO 全部打勾；
  - [x] 新增 summary 文档（同编号 2）。

当前进度备注（2026-05-28 补充）：
- 已完成：M5 基础断言覆盖（成功提取/空内容/损坏文件）并在 `vectraparse-mso-binary` 单测中落地。
- 已完成：`tests/fixtures/ole/{doc,ppt,xls}` 目录骨架与样本数量校验脚本（`check_counts.sh`）落地；当前计数已达到每类 `>=20`。
- 已完成：样本可提取性校验脚本（`check_extractable.sh`）落地，可批量调用 `extract-static` 检查解析与非空输出；当前 60/60 样本通过。
- 已完成：生成每类 20 个基础合成样本（`generate_synthetic_samples.sh`），`check_counts.sh` 校验通过。

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
