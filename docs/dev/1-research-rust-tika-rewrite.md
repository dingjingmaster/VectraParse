# Rust 重写 Tika 内容与文件类型提取调研报告

> 文档元数据
> - 文件编号：1
> - 文档类型：research
> - 文件路径：docs/dev/1-research-rust-tika-rewrite.md
> - 文档版本：v1.0.0
> - 最后更新：2026-05-27
> - 关联需求：分析 `/data/source/tika` 项目结构，面向 Rust 静态库和动态库重写内容提取与文件类型提取能力；忽略 `tika-server` 等服务/应用层项目。

## 1. 问题与边界

- 问题描述：Apache Tika 是 Java/Maven 多模块项目，本次目标不是复用 JVM，也不是重写 server/app/batch 工具，而是提炼其内容提取、元数据提取、文件类型识别能力，设计可由其他项目链接调用的 Rust 库。
- 调研目的：梳理 Tika 模块结构、核心调用链、格式覆盖和重写风险，给下一阶段 Rust 架构与里程碑计划提供依据。
- 包含：`tika-core`、`tika-parsers`、与内容/元数据序列化相关的 `tika-serialization`、XMP 转换相关的 `tika-xmp`，以及可作为测试素材/行为规格参考的 parser/core 测试资源。
- 不包含：`tika-server`、`tika-app`、`tika-batch`、`tika-eval`、`tika-fuzzing`、`tika-deployment`、`tika-dotnet`、示例项目、服务脚本、CLI/REST/批处理封装。
- 非目标：不按 Java API 原样移植，不承诺一次性覆盖 Tika 全部 75 个 SPI parser，不把 OCR、NER、翻译、深度学习、外部命令执行作为首版核心能力。
- 禁止触碰范围：本阶段只读分析 `/data/source/tika`，不修改该仓库；当前 VectraParse 仓库只新增调研文档和索引。

## 2. 当前证据

- 本地 Tika 版本与工作树：`/data/source/tika` 在 `master` 分支，提交 `618345263ee41108e1a225dbcdbb8db16b2aae28`，工作树干净；根 POM 版本为 `2.0.0-SNAPSHOT`，Java 编译目标为 1.8。
- 顶层模块：根 `pom.xml` 聚合 `tika-parent`、`tika-core`、`tika-parsers`、`tika-bundle`、`tika-xmp`、`tika-serialization`、`tika-batch`、`tika-app`、`tika-server`、`tika-fuzzing`、`tika-translate`、`tika-langdetect`、`tika-example`、`tika-java7`、`tika-eval`、`tika-dl`、`tika-nlp`。
- 代码规模：全仓约 2565 个文件；`tika-core/src/main/java` 约 256 个 Java 文件；`tika-parsers/src/main/java` 约 355 个 Java 文件。
- SPI 规模：`tika-parsers/src/main/resources/META-INF/services/org.apache.tika.parser.Parser` 注册 75 个 parser；`org.apache.tika.detect.Detector` 注册 4 个 detector；`EncodingDetector` 注册 3 个编码检测器。
- MIME 数据库：`tika-core/src/main/resources/org/apache/tika/mime/tika-mimetypes.xml` 含约 1599 个 `mime-type`、355 个 `magic`、1302 个 `glob`、62 个 `root-XML`、141 个 `alias`、321 个 `sub-class-of`。
- Bug 证据等级：不适用。本阶段是需求调研，不是 bug 修复。
- 证据不足项：未运行 Tika 测试套件，未做格式级提取质量基准；Rust crate 选型只做初步生态核对，必须在 Plan/POC 阶段用真实样本验证。

## 3. 安全门禁摘要

| 项 | 结论 |
|----|------|
| 风险矩阵初判 | L4，大型重写，涉及跨语言 ABI、解析不可信输入、构建发布和长期架构 |
| 命令权限 | 已执行 C0 只读扫描；新增文档属于 C1 工作区写入 |
| 高风险开发门禁 | 是，后续会涉及 Rust FFI/ABI、内存边界、构建链接、文件解析安全 |
| 破坏性操作 | 否 |
| 用户已有修改 | 当前仓库已有未跟踪 `.codex`，本次不触碰 |
| 用户确认事项 | 下一阶段需确认首批格式优先级、目标平台、是否允许 native 依赖、C ABI 输出形态 |

## 4. Tika 项目结构与核心调用链

### 4.1 模块职责

- `tika-core`：核心 API 与基础设施，包含 `Tika` facade、`TikaConfig`、`Detector`、`MimeTypes`、`Parser`、`AutoDetectParser`、`CompositeParser`、`Metadata`、`ParseContext`、SAX handler、临时资源和递归嵌入文档处理。
- `tika-parsers`：具体格式 parser 与容器 detector，覆盖 PDF、Office/OLE/OOXML/ODF、HTML/XML/Text/CSV、压缩包、邮件、图片、音频、视频、EPUB、科学/地理数据、OCR、外部命令等。
- `tika-serialization`：主要处理 `Metadata`/`MetadataList` JSON 序列化，可借鉴 Rust FFI 结果 JSON 输出。
- `tika-xmp`：把 Tika metadata 转为 XMP，首版 Rust 可不做，作为后续 metadata 标准化增强项。
- `tika-langdetect`、`tika-translate`、`tika-nlp`、`tika-dl`：属于内容后处理或模型/服务能力，建议首版排除。
- `tika-app`、`tika-server`、`tika-batch`、`tika-eval`、`tika-fuzzing`、`tika-dotnet`、`tika-deployment`：运行形态、批处理、评估、测试工具或平台封装，不进入 Rust 核心库范围。

### 4.2 识别链路

- `Tika.detect(...)` 是 facade；流式输入会保证 mark/reset 能力，然后委托 `Detector.detect`。
- `TikaConfig` 默认构建 `MimeTypes`、`DefaultDetector`、`DefaultEncodingDetector`、`DefaultParser`，并支持 XML 配置覆盖。
- `DefaultDetector` 从 Java ServiceLoader 加载 detector，排序后把 `MimeTypes` 作为最终 fallback。
- `CompositeDetector` 逐个执行 detector，选择当前结果的更具体 specialization；遇到 content-type override 时可短路。
- `MimeTypes.detect` 的关键顺序是：读取最多 64 KiB magic header、按 magic priority 匹配、对 XML/HTML 进一步按 root element 细分、无 magic 时回退文本检测，再结合文件名 `RESOURCE_NAME_KEY` 和 `CONTENT_TYPE` hint 进一步 specialize。
- `tika-parsers` 额外提供容器 detector：例如 OLE/POIFS、ZIP/OOXML/ODF/iWork、binary plist。Rust 不能只靠 magic/glob，否则 OOXML、ODF、APK、JAR、iWork 等容器格式会大量误判为 zip。

### 4.3 内容提取链路

- `AutoDetectParser.parse` 把输入包装为 `TikaInputStream`，自动检测 MIME，写入 `Metadata.CONTENT_TYPE`，检查空文件，套 `SecureContentHandler`，并设置默认 `ParsingEmbeddedDocumentExtractor`。
- `CompositeParser` 根据 `Metadata.CONTENT_TYPE` 和 `MediaTypeRegistry` 的 supertype 关系查找 parser；找不到时使用 fallback。
- Parser 输出不是纯字符串，而是 XHTML SAX events；`BodyContentHandler`、`ToTextContentHandler`、`ToHTMLContentHandler`、`ToXMLContentHandler` 再把 SAX 转为正文、HTML 或 XML。
- 嵌入文档通过 `EmbeddedDocumentExtractor` 递归处理；`RecursiveParserWrapper` 把主文档和嵌入文档都转成一组 `Metadata`，并记录 embedded path、解析耗时、异常、写入限制等。
- Rust 侧建议直接建结构化结果模型，保留 Tika 的递归嵌入文档语义，但不要复刻 SAX 作为外部 API。

## 5. Parser 覆盖与 Rust 重写优先级

| 优先级 | 格式族 | Tika 对应模块/类族 | Rust 侧建议 |
|--------|--------|---------------------|-------------|
| P0 | MIME/文件类型识别 | `tika-core/mime`、`DefaultDetector`、`MimeTypes`、ZIP/OLE detector | 先移植 MIME 数据库和 detector 调度；`infer`/`mime_guess` 只能做辅助，不能替代 Tika 的 specialization 规则 |
| P0 | 纯文本、CSV、HTML、XML | `txt`、`csv`、`html`、`xml`、encoding detector | 用 `encoding_rs`/`chardetng`、`csv`、`html2text`、`quick-xml` 做首批闭环 |
| P1 | ZIP 容器与 Office Open XML/ODF/EPUB | `pkg`、`microsoft.ooxml`、`opendocument`、`epub` | 用 `zip` + 自研关系/manifest/content-types 解析；表格可评估 `calamine`，DOCX/PPTX 多半需要自研文本抽取层 |
| P1 | PDF | `pdf/PDFParser` 基于 PDFBox | `pdf-extract`/`lopdf` 可做 POC，但要单独验证加密、编码、布局和异常 PDF；PDF 是质量风险最高的首批格式 |
| P2 | 邮件与附件 | `mail`、`mbox`、`OutlookPSTParser` | RFC822 可评估 `mail-parser`；PST/MSG 需要专项调研 |
| P2 | 图片/音视频 metadata | `image`、`jpeg`、`mp3`、`mp4`、`audio` | 可评估 `image`、`nom-exif`、`lofty`；首版以 metadata 为主，不做 OCR |
| P3 | 旧 Office/OLE | `microsoft/OfficeParser`、POI、POIFS | 可用 `cfb` 读取容器，但 DOC/PPT 二进制格式文本抽取工作量大，应后置或限定范围 |
| P3 | 压缩包递归 | `pkg/PackageParser`、`CompressorParser`、`RarParser` | 支持 zip/tar/gzip/bzip2/xz/zstd 优先；rar/7z 按依赖许可与安全性再定 |
| P4 | OCR、外部命令、NLP、地理/科学格式 | `ocr`、`external`、`ner`、`gdal`、`netcdf`、`grib` | 首版排除，后续用 feature gate 和显式配置启用 |

## 6. 候选方案

| 方案 | 核心思路 | 优点 | 风险/代价 | 适用条件 |
|------|----------|------|-----------|----------|
| A：Java Tika 行为兼容型 Rust 重写 | 移植 Tika core 模型与 MIME 数据库，逐步替换 parser | 行为边界清楚，容易用 Tika 测试样本做差异验证；适合长期库化 | 工作量大，PDF/Office 质量追平困难 | 目标是长期替代 Tika，且允许分阶段覆盖格式 |
| B：Rust crate 聚合型快速库 | 组合现有 Rust crate，API 只承诺最佳努力提取 | 初期速度快，能较快产出静态/动态库 | 行为与 Tika 差异大，格式检测和 metadata 一致性弱 | 目标只是通用提取，不强调 Tika 兼容 |
| C：GraalVM/native Tika 包装 | 用 Rust FFI 包一层 native Tika 或现有 Java 逻辑 | 短期功能覆盖最强 | 不是真正 Rust 重写；构建复杂，体积和依赖重，静态库交付困难 | 仅适合过渡验证，不符合当前“以 Rust 重写”目标 |

## 7. 推荐结论

- 推荐方案：选 A，但实现节奏采用“Rust core + 格式族插件”的分阶段策略；禁止把 75 个 Java parser 作为首版范围。
- 取舍理由：用户目标是 Rust 静态库/动态库供其他项目使用，关键风险在 ABI 稳定、解析安全和可控依赖；先复刻 Tika 的调度/结果/检测模型，比先堆 parser 更能支撑长期扩展。
- 需要进入 Plan 的关键约束：
  - Cargo 输出建议同时包含 `rlib`、`staticlib`、`cdylib`；C ABI 用 `extern "C"`、opaque handle、显式 free 函数，避免跨 ABI 传 Rust `String`/`Vec`/trait object。
  - C ABI 首版建议返回 UTF-8 JSON 结果，字段包含 `mime_type`、`metadata`、`content`、`embedded`、`warnings`、`error`，保留多值 metadata。
  - 所有 parser 必须有资源限制：输入大小、解压比例、递归深度、嵌入文件数量、输出字符上限、超时/取消、临时文件策略。
  - native 依赖必须 feature-gate；默认包尽量 pure Rust，避免静态库链接被系统库污染。
  - MIME 数据库应从 `tika-mimetypes.xml` 转为构建期生成的 Rust 表或压缩资源，保留 magic/glob/rootXML/subtype/alias 规则。
- 需要用户确认的问题：
  - 首批必须支持哪些格式：PDF、DOCX、XLSX、PPTX、HTML、TXT/CSV、ZIP/EPUB/ODF 是否全部进 MVP？
  - 目标平台：Linux only，还是 Linux/macOS/Windows；是否需要 musl 静态链接？
  - 是否允许依赖 C/C++/系统库，例如 PDFium、Poppler、Tesseract、libmagic？
  - 对 Tika 兼容性的要求：MIME 类型/metadata key/content 文本是否需要与 Tika 回归对齐？
  - FFI 消费方语言：C/C++、Go、Python、Java/JNI、Node，决定 header、符号命名和内存释放约定。
- 后续验证方向：
  - 从 Tika `tika-core` 和 `tika-parsers` 测试资源抽样建立 golden corpus。
  - 首先验证 MIME detection：扩展名、magic、XML root、ZIP/OLE 容器 specialization。
  - 再验证提取：TXT/CSV/HTML/XML、DOCX/XLSX/EPUB/ODF、PDF 的文本、metadata、嵌入文档。
  - 对不可信输入建立 fuzzing 与 resource-limit 测试，尤其是 zip bomb、深递归 XML、损坏 PDF/Office。

## 8. Research 阶段审视

- 安全审查员：已按 L4 处理；本阶段只读 Tika、只新增当前任务文档；后续实现将命中 FFI/ABI、解析不可信输入、构建链接门禁，必须在 Plan 中明确 C ABI、内存释放、panic 边界、资源限制和回滚策略。
- 高级产品：需求边界已裁剪为内容/文件类型提取库；server/app/batch/eval/OCR/NLP/翻译等未纳入首版，避免范围失控。
- 高级架构师：推荐先做 core 和 registry，再做 parser 插件，避免把 Java 多模块和 ServiceLoader 模型照搬到 Rust；静态/动态库交付会反向约束依赖选择。
- 高级工程师：调研尚未包含运行时基准和 crate POC，不能直接进入编码；下一阶段应先产出格式优先级和最小 API，再做 POC。

## 9. 参考资料

- 本地源码：`/data/source/tika/pom.xml`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/Tika.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/config/TikaConfig.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/detect/DefaultDetector.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/mime/MimeTypes.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/parser/AutoDetectParser.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/parser/CompositeParser.java`
- 本地源码：`/data/source/tika/tika-core/src/main/resources/org/apache/tika/mime/tika-mimetypes.xml`
- Rust Reference：`https://doc.rust-lang.org/stable/reference/linkage.html`
- docs.rs `infer`：`https://docs.rs/infer/latest/infer/`
- docs.rs `mime_guess`：`https://docs.rs/mime_guess/`
- docs.rs `quick-xml`：`https://docs.rs/quick-xml`
- docs.rs `zip`：`https://docs.rs/zip/latest/zip/`
- docs.rs `calamine`：`https://docs.rs/calamine/latest/calamine/`
- docs.rs `pdf-extract`：`https://docs.rs/pdf-extract/latest/`
- docs.rs `lopdf`：`https://docs.rs/lopdf/latest/lopdf/`
- docs.rs `mail-parser`：`https://docs.rs/mail-parser/latest/mail_parser/`
- docs.rs `cfb`：`https://docs.rs/cfb`
- docs.rs `chardetng`：`https://docs.rs/chardetng/`
- docs.rs `html2text`：`https://docs.rs/html2text/latest/html2text/`
- docs.rs `nom-exif`：`https://docs.rs/nom-exif`
- docs.rs `lofty`：`https://docs.rs/lofty/latest/lofty/`
