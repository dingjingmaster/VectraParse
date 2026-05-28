# OLE 二进制 Office（DOC/PPT/XLS）提取调研

> 文档元数据
> - 文件编号：2
> - 文档类型：research
> - 文件路径：docs/dev/2-research-ole-doc-ppt-xls-extraction.md
> - 文档版本：v1.0.0
> - 最后更新：2026-05-28
> - 关联需求：基于 `/data/source/libreoffice-core/core` 调研并给出 `.doc/.ppt/.xls` 内容提取可行方案；约束为不引入新的系统二进制依赖。

## 1. 结论（先给答案）

- **可行**：可以在当前 Rust 项目内实现 `.doc/.ppt/.xls` 文本提取，且不引入新的系统二进制依赖。
- **前提**：采用“**纯 Rust 重写解析链路** + 参考 LibreOffice 实现与 MS 二进制规范”，不直接嵌入 LibreOffice 代码或其构建产物。
- **范围建议**：
  1. Phase-1 先达成可用正文提取（主文档/主幻灯片/单元格可见值）。
  2. Phase-2 再补格式细节（注释、脚注、页眉页脚、公式结果、更多编码边界）。

## 2. 现状与差距

当前 `VectraParse` 对这三类老格式主要依赖 OLE 头识别和字符串扫描，未按格式结构化解析：

- 当前实现入口：`crates/vectraparse-parsers/src/lib.rs` 中 `OleLegacyParser` / `MsSpecialParser` / `LegacyDocParser`。
- 表现：能识别部分流名，但正文常混杂骨架字段、乱码或空内容。

这与目标“稳定提取 `.doc/.ppt/.xls` 真实文本内容”不一致。

## 3. LibreOffice 源码证据（可复用思路）

### 3.1 DOC（WW8）

- 关键目录：`/data/source/libreoffice-core/core/sw/source/filter/ww8`
- 关键线索：
  - `ww8scan.hxx` 中定义了 Word OLE 常见流名：`1Table`、`0Table`、`Data` 等。
  - `ww8scan.cxx` 中 `WW8Fib::WW8Fib(...)` 注释与实现明确按 FIB 读取流程处理 Word 二进制（含版本、标志、偏移）。
  - 代码中大量 `WW8PLCF*` / `WW8PLCFpcd*` / piece table 相关结构，说明正文提取核心是 **CLX/Piece Table**，而不是直接扫字符串。

### 3.2 PPT（PPT Binary）

- 关键目录：`/data/source/libreoffice-core/core/sd/source/filter/ppt`、`.../sdpptwrp.cxx`
- 关键线索：
  - `sdpptwrp.cxx` 在 OLE 存储中打开 `"PowerPoint Document"` 流进行导入。
  - `pptin.cxx` 使用 `"Current User"`、`SeekToRec(...PPT_PST_Document...)`、`TextBytesAtom` / `TextCharsAtom` / `CString` 等记录体系读取文本。
  - 说明提取路径应是：**CurrentUser/UserEdit/Persist 链 -> 记录遍历 -> 文本 atom 收集**。

### 3.3 XLS（BIFF）

- 关键目录：`/data/source/libreoffice-core/core/sc/source/filter/excel`
- 关键线索：
  - `impop.cxx` 的 `ImportExcel::ImportExcel` 与 `Read()` 显示 BIFF 状态机导入流程（BOF/EOF、sheet 子流、record 驱动）。
  - `read.cxx` 明确以 record 流推进并按 BIFF 版本分支处理。
  - 说明文本提取应基于 **Workbook 流 BIFF records**（SST/LABELSST/LABEL/NUMBER/RK/FORMULA cached result 等）而非字符串扫取。

## 4. 不引入新二进制依赖的实现方案

## 4.1 约束解释

- “不引入新的二进制依赖”理解为：
  - 不新增系统命令依赖（`soffice`、`catdoc`、`antiword` 等）。
  - 不依赖额外动态库/系统包。
  - 可使用纯 Rust crate（源码编译进现有 `staticlib/cdylib`）。

### 4.2 推荐实现架构

- 新增 `crates/vectraparse-mso-binary`（纯 Rust）：
  - `ole`：CFB/OLE 存储读取（目录、流、sector 链）。
  - `doc`：FIB + CLX/Piece Table + 编码解码 + 文本拼接。
  - `ppt`：PPT record parser（Document container + Text atoms）。
  - `xls`：BIFF record parser（Workbook/Worksheet，SST + cell records）。
- 在 `vectraparse-parsers` 中替换现有 `OleLegacyParser` 文本提取路径。

## 4.3 各格式最小可行提取（MVP）

1. **DOC**
   - 必要流：`WordDocument` + `0Table/1Table`。
   - 必要结构：FIB、`fcClx/lcbClx`、Piece Table（PCD）。
   - 输出：主文档正文（按 piece 顺序拼接，处理 Unicode/ANSI piece）。

2. **PPT**
   - 必要流：`PowerPoint Document`（可选 `Current User`）。
   - 必要结构：record header + container 遍历；提取 `TextBytesAtom`、`TextCharsAtom`、`CString`。
   - 输出：按 slide 顺序聚合文本。

3. **XLS**
   - 必要流：`Workbook`（或老版本 `Book`）。
   - 必要结构：BOF/EOF、BOUNDSHEET、SST、LABELSST/LABEL/NUMBER/RK/FORMULA。
   - 输出：每个 sheet 的可见 cell 文本（可按行列拼接）。

## 5. 风险与边界

- DOC 风险最高：Piece Table、编码与版本兼容复杂，需严格对齐 FIB/CLX 偏移。
- PPT 次高：record 链复杂，需防御损坏文件与递归 container。
- XLS 相对可控：BIFF record 体系稳定，但版本差异（BIFF2~8）需分阶段覆盖。

建议先以 **BIFF8 / Word97+ / PPT97+** 为主线，老版本逐步兜底。

## 6. 验收标准（确保“能提取”）

### 6.1 功能验收

- `.doc`：输出正文不再仅是 OLE 骨架字段（`Root Entry/WordDocument/...`），应包含真实段落文本。
- `.ppt`：输出 slide 文本（标题/正文）而不是仅 metadata/atom 名称。
- `.xls`：输出单元格值（文本/数字缓存值），可区分 sheet。

### 6.2 回归样例建议

- 每种格式至少 20 个样例：
  - 中文/英文/混合编码；
  - 大文件、损坏边界、包含嵌入对象；
  - 带公式与共享字符串（xls）、带注释与页眉（doc）、多母版与备注（ppt）。

### 6.3 与当前 CLI 验收

- `./target/extract-static <file>` 在这三类输入上 `Content` 必须非空且可读。

## 7. 实施建议（下一步）

1. 先做 `doc` 主链路（价值最高，当前痛点最明显）。
2. 并行推进 `xls` 基础 BIFF 提取（SST + LABELSST）。
3. 最后补 `ppt` 文本 atoms 聚合。
4. 每完成一个里程碑都提交，并更新 `docs/dev` 计划状态。

---

本调研结论：在“不引入新的系统二进制依赖”的约束下，依照 LibreOffice 的解析思路重写是可行且推荐的路线；三者都能做到稳定文本提取，但应分阶段落地，避免一次性全量复杂度。
