# Rust 重写 Tika 内容与文件类型提取调研报告

> 文档元数据
> - 文件编号：1
> - 文档类型：research
> - 文件路径：docs/dev/1-research-rust-tika-rewrite.md
> - 文档版本：v1.0.2
> - 最后更新：2026-05-27
> - 关联需求：分析 `/data/source/tika` 项目结构，面向 Rust 静态库和动态库重写内容提取与文件类型提取能力；忽略 `tika-server` 等服务/应用层项目。

## 1. 问题与边界

- 问题描述：Apache Tika 是 Java/Maven 多模块项目，本次目标不是复用 JVM，也不是重写 server/app/batch 工具，而是提炼其内容提取、元数据提取、文件类型识别能力，设计可由其他项目链接调用的 Rust 库。
- 调研目的：梳理 Tika 模块结构、核心调用链、格式覆盖和重写风险，给下一阶段 Rust 架构与里程碑计划提供依据。
- 包含：`tika-core`、`tika-parsers`、`tika-xmp`、`tika-serialization`、`tika-langdetect`、`tika-translate`、`tika-nlp`、`tika-dl`、`tika-java7` 中与文件类型识别、内容提取、元数据提取、嵌入文档处理、OCR、语言识别、翻译、NLP、深度学习增强和结果序列化相关的能力，以及可作为测试素材/行为规格参考的资源。
- 不包含：`tika-server`、`tika-app`、`tika-batch`、`tika-eval`、`tika-fuzzing`、`tika-deployment`、`tika-dotnet`、示例项目、服务脚本、CLI/REST/批处理封装；其中测试素材、格式样本和行为预期可作为验证输入引用。
- 非目标：不在 Research 阶段裁掉主要功能，也不在 Research 阶段决定哪些相关功能延后；所有与主题相关的 Tika 能力先纳入总体规划，具体实现顺序、里程碑、依赖策略和风险接受度在 Plan 阶段排序。
- 禁止触碰范围：本阶段只读分析 `/data/source/tika`，不修改该仓库；当前 VectraParse 仓库只新增调研文档和索引。

## 2. 当前证据

- 本地 Tika 版本与工作树：`/data/source/tika` 在 `master` 分支，提交 `618345263ee41108e1a225dbcdbb8db16b2aae28`，工作树干净；根 POM 版本为 `2.0.0-SNAPSHOT`，Java 编译目标为 1.8。
- 顶层模块：根 `pom.xml` 聚合 `tika-parent`、`tika-core`、`tika-parsers`、`tika-bundle`、`tika-xmp`、`tika-serialization`、`tika-batch`、`tika-app`、`tika-server`、`tika-fuzzing`、`tika-translate`、`tika-langdetect`、`tika-example`、`tika-java7`、`tika-eval`、`tika-dl`、`tika-nlp`。
- 代码规模：全仓约 2565 个文件；`tika-core/src/main/java` 约 256 个 Java 文件；`tika-parsers/src/main/java` 约 355 个 Java 文件。
- SPI 规模：`tika-parsers/src/main/resources/META-INF/services/org.apache.tika.parser.Parser` 注册 75 个 parser；`org.apache.tika.detect.Detector` 注册 4 个 detector；`EncodingDetector` 注册 3 个编码检测器。
- Parser 包目录：`tika-parsers/src/main/java/org/apache/tika/parser` 下有 59 个一级包目录，其中部分能力没有出现在 SPI parser 注册文件中，但仍属于 Tika 功能面或可配置增强能力。
- MIME 数据库：`tika-core/src/main/resources/org/apache/tika/mime/tika-mimetypes.xml` 含约 1599 个 `mime-type`、355 个 `magic`、1302 个 `glob`、62 个 `root-XML`、141 个 `alias`、321 个 `sub-class-of`。
- Bug 证据等级：不适用。本阶段是需求调研，不是 bug 修复。
- 证据不足项：未运行 Tika 测试套件，未做格式级提取质量基准；Rust crate 选型只做初步生态核对，必须在 Plan/POC 阶段用真实样本验证。已复查出当前文档仍有若干 Tika 功能族表达过粗或未显式列出，见第 5.1 节。

## 3. 安全门禁摘要

| 项 | 结论 |
|----|------|
| 风险矩阵初判 | L4，大型重写，涉及跨语言 ABI、解析不可信输入、构建发布和长期架构 |
| 命令权限 | 已执行 C0 只读扫描；新增文档属于 C1 工作区写入 |
| 高风险开发门禁 | 是，后续会涉及 Rust FFI/ABI、内存边界、构建链接、文件解析安全 |
| 破坏性操作 | 否 |
| 用户已有修改 | 当前仓库已有未跟踪 `.codex`，本次不触碰 |
| 用户确认事项 | 下一阶段需确认规划顺序、目标平台、是否允许 native 依赖、C ABI 输出形态 |

## 4. Tika 项目结构与核心调用链

### 4.1 模块职责

- `tika-core`：核心 API 与基础设施，包含 `Tika` facade、`TikaConfig`、`Detector`、`MimeTypes`、`Parser`、`AutoDetectParser`、`CompositeParser`、`Metadata`、`ParseContext`、SAX handler、临时资源和递归嵌入文档处理。
- `tika-parsers`：具体格式 parser 与容器 detector，覆盖 PDF、Office/OLE/OOXML/ODF、HTML/XML/Text/CSV、压缩包、邮件、图片、音频、视频、EPUB、科学/地理数据、OCR、外部命令等。
- `tika-serialization`：主要处理 `Metadata`/`MetadataList` JSON 序列化，可借鉴 Rust FFI 结果 JSON 输出。
- `tika-xmp`：把 Tika metadata 转为 XMP，属于元数据标准化输出能力，纳入总体规划。
- `tika-langdetect`、`tika-translate`、`tika-nlp`、`tika-dl`：属于内容识别、内容增强和模型能力，纳入总体规划；Plan 阶段再决定与基础提取链路的集成顺序和可选依赖边界。
- `tika-java7`：提供 Java NIO FileTypeDetector 等文件类型识别入口，作为 Rust 文件类型识别 API 形态参考。
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

## 5. 功能覆盖范围与 Plan 排序依据

> 本节不做“做/不做”裁剪；所有行都纳入 Rust 重写总体规划。`规划排序依据` 只用于后续 Plan 阶段安排先后、拆里程碑和验证策略。

| 功能族 | Tika 对应模块/类族 | Rust 侧规划要点 | Plan 排序依据 |
|--------|---------------------|------------------|---------------|
| MIME/文件类型识别 | `tika-core/mime`、`DefaultDetector`、`MimeTypes`、ZIP/OLE detector、`tika-java7` FileTypeDetector | 移植 MIME 数据库和 detector 调度；`infer`/`mime_guess` 只能做辅助，不能替代 Tika 的 specialization 规则 | 所有解析入口依赖它，应作为架构基础 |
| 高级检测模型与 override | `OverrideDetector`、`ZeroSizeFileDetector`、`NameDetector`、`TypeDetector`、`TrainedModelDetector`、`NNExampleModelDetector` | 规划 content-type override、空文件检测、文件名/显式类型 hint、训练模型检测和神经网络示例模型的 Rust 等价能力 | 影响 MIME 准确率和兼容性，需与基础检测合并设计 |
| 纯文本、CSV、HTML、XML | `txt`、`csv`、`html`、`xml`、encoding detector | 规划编码检测、文本归一化、结构化文本和正文抽取；可评估 `encoding_rs`/`chardetng`、`csv`、`html2text`、`quick-xml` | 覆盖面广，可作为提取结果模型验证基线 |
| 文本衍生格式与源码 | `code`、`strings`、`iptc`、`xliff`、`feed`、`dif`、`envi` | 规划源码高亮/语言识别、二进制 strings 提取、IPTC ANPA、XLIFF/XLZ、本地化文本、RSS/Atom feed、DIF、ENVI header 等文本型格式 | 这些格式容易被纯文本兜底吞掉，Plan 中需单独验收 |
| PDF | `pdf/PDFParser`、PDF OCR 集成 | 规划文本、metadata、附件/嵌入资源、图片 OCR 策略；可评估 `pdf-extract`、`lopdf`、PDFium/Poppler 等路线 | 价值高、风险高，需要专项 POC 和质量基准 |
| Office Open XML/ODF/EPUB/iWork | `pkg`、`microsoft.ooxml`、`opendocument`、`epub`、`iwork` | 规划 ZIP 容器 specialization、关系文件、manifest、正文、metadata、嵌入文档；表格可评估 `calamine` | 常见办公文档主链路，需与容器解析同步设计 |
| 旧 Office/OLE 与微软专有格式 | `microsoft/OfficeParser`、`POIFSContainerDetector`、`JackcessParser`、`OneNoteParser`、`TNEFParser`、`EMFParser`、`WMFParser`、`MSOwnerFileParser`、`OldExcelParser` | 规划 CFB/OLE 容器、DOC/XLS/PPT/MSG/Access/OneNote/TNEF/EMF/WMF/owner file/旧 Excel 等文本和 metadata 提取；可评估 `cfb` | 格式复杂、样本差异大，需要专项分解 |
| 其他办公与文档格式 | `rtf`、`wordperfect`、`hwp`、`chm`、`apple`、`odf/OpenDocumentContentParser`、`opendocument/OpenOfficeParser` | 规划 RTF、WordPerfect、Quattro Pro、HWP、CHM、AppleSingle/PList、OpenOffice/ODF 子解析器等能力 | 不应被旧 Office 或文本格式大类吞掉，需要专门样本 |
| 压缩包与递归嵌入文档 | `pkg/PackageParser`、`CompressorParser`、`RarParser` | 规划 zip/tar/gzip/bzip2/xz/zstd/rar/7z 等容器/压缩流识别、递归解析、解压限制和嵌入路径 | 直接影响嵌入文档能力和安全边界 |
| 邮件、邮箱与附件 | `mail`、`mbox`、`OutlookPSTParser` | 规划 RFC822/MIME、MBOX、PST/MSG、附件递归和 charset 处理；可评估 `mail-parser` | 与递归嵌入文档共享结果模型 |
| 图片、音频、视频 metadata | `image`、`jpeg`、`mp3`、`mp4`、`audio`、`video`、`captioning` | 规划 EXIF/XMP/IPTC、音频标签、视频容器 metadata、图片格式特化、图像/视频 captioning、必要时的缩略图/嵌入资源；可评估 `image`、`nom-exif`、`lofty` | metadata 覆盖面广，依赖选择影响发布体积 |
| OCR 与外部解析器 | `ocr/TesseractOCRParser`、`external/ExternalParser`、PDF OCR 集成 | 规划外部命令/系统库调用策略、超时、沙箱、可选依赖、错误降级和跨平台配置 | 风险高但属于功能范围，Plan 中明确启用条件 |
| 语言识别、翻译、NLP、NER、医学文本、深度学习 | `tika-langdetect`、`tika-translate`、`tika-nlp`、`tika-dl`、`ner`、`sentiment`、`recognition`、`ctakes` | 规划为内容增强层，定义语言识别、翻译服务、NER、情感分析、对象识别、医学文本 cTAKES、模型资源、可选依赖和失败降级 | 与核心提取解耦，但纳入总体库能力 |
| 数据库、表格与专业数据格式 | `jdbc`、`dbf`、`isatab`、`gdal`、`netcdf`、`grib`、`hdf`、`mat`、`sas`、`dwg`、`geoinfo`、`geo`、`pot`、`prt` | 规划 SQLite/JDBC、DBF、ISA-Tab、GDAL、NetCDF、GRIB、HDF、MAT、SAS、DWG、地理信息、地理主题、Pooled Time Series、PRT 等 metadata/文本提取 | 依赖和格式复杂度高，需要独立验收 |
| 加密、签名与安全相关格式 | `crypto/Pkcs7Parser`、`crypto/TSDParser`、加密 PDF/Office、`PasswordProvider` | 规划 PKCS#7、timestamp、密码/加密文档处理、权限 metadata 和错误降级 | 涉及安全、合规和错误模型，不能只按普通 parser 处理 |
| Java/二进制与可执行文件 | `asm/ClassParser`、`executable/ExecutableParser`、`strings` | 规划 Java class、ELF/PE、Mach-O 或平台可执行文件 metadata、二进制 strings 提取 | 与安全扫描和文件类型识别关联强 |
| 字体格式 | `font/AdobeFontMetricParser`、`font/TrueTypeParser` | 规划 AFM、TTF/OTF 等字体 metadata 和命名表提取 | 单独格式族，当前不应被图片/文档大类覆盖 |
| XMP 与序列化输出 | `tika-xmp`、`tika-serialization` | 规划多值 metadata、XMP 映射、JSON/结构化输出、C ABI 内存释放约定 | 直接影响动态库/静态库对外使用体验 |
| 内容处理器与抽取基础设施 | `sax`、`sax/xpath`、`extractor`、`embedder`、`parser/digest`、`parser/multiple`、`fork`、`NetworkParser` | 规划正文/HTML/XML 输出、链接/电话/标准号抽取、XPath 过滤、ContainerExtractor、Embedder、digest/hash、fallback/supplementing parser、fork 隔离、网络解析器等运行时基础能力 | 这些不是单一格式 parser，但决定 Tika 行为兼容度和库 API 形态 |

### 5.1 与 Tika 对照后的当前遗漏

- Parser 目录仍需显式跟踪：`apple`、`asm`、`captioning`、`chm`、`code`、`crypto`、`ctakes`、`dbf`、`dif`、`envi`、`executable`、`feed`、`font`、`hwp`、`isatab`、`jdbc`、`journal`、`multiple`、`pot`、`prt`、`rtf`、`strings`、`wordperfect`、`xliff`。这些在 v1.0.1 中多被隐含在大类里，Plan 阶段容易漏掉。
- SPI 注册 parser 中未逐项落表的能力：AppleSingle/PList、Java class、PKCS#7/TSD、DWG、ENVI、executable、feed、font、HWP、IPTC ANPA、MBOX/PST、EMF/WMF、MS owner file、旧 Excel、Word 2006 XML、SpreadsheetML、NetCDF/HDF/GRIB/MAT/SAS、RTF、WordPerfect/Quattro Pro、DIF、CHM、source code、SQLite、ISA-Tab、Grobid/Journal、ICNS、XLIFF/XLZ。
- 非 SPI 但和 Tika 功能面相关的 parser/增强能力：captioning TensorFlow REST、object recognition、cTAKES、NER 多后端、sentiment、strings/Latin1 strings、PDF preflight、RTF object data、fallback/supplementing parser、digest parser、external parser XML 配置。
- Core 层遗漏：`TrainedModelDetector`/`NNExampleModelDetector`、`LanguageIdentifier` n-gram 资源、`PasswordProvider`、`ContainerExtractor`/`ParserContainerExtractor`、`Embedder`/`ExternalEmbedder`、`LinkContentHandler`、`PhoneExtractingContentHandler`、`StandardsExtractingContentHandler`、`XMPContentHandler`、`XPathParser`、`fork` 隔离能力和 `NetworkParser`。
- 元数据命名空间遗漏：`AccessPermissions`、`Database`、`Font`、`Geographic`、`IPTC`、`Office`、`OfficeOpenXML*`、`PDF`、`PagedText`、`Photoshop`、`QuattroPro`、`RTFMetadata`、`TIFF`、`WordPerfect`、`XMP*`、`XMPDM` 等 key 集合需要作为 Rust metadata schema 对照输入。
- 验证资产遗漏：`tika-eval`、`tika-fuzzing`、各模块测试资源不属于库运行时功能，但应进入 Plan 的 golden corpus、差异测试、fuzzing 和压力测试规划。

## 6. 候选方案

| 方案 | 核心思路 | 优点 | 风险/代价 | 适用条件 |
|------|----------|------|-----------|----------|
| A：Java Tika 行为兼容型 Rust 重写 | 移植 Tika core 模型与 MIME 数据库，把与主题相关的 Tika parser 和增强能力全部纳入总体规划，再在 Plan 阶段排序实施 | 行为边界清楚，容易用 Tika 测试样本做差异验证；适合长期库化 | 工作量大，PDF/Office/OCR/NLP/专业格式质量追平困难 | 目标是长期替代 Tika，且需要覆盖主要功能 |
| B：Rust crate 聚合型快速库 | 组合现有 Rust crate，API 只承诺最佳努力提取 | 初期速度快，能较快产出静态/动态库 | 行为与 Tika 差异大，格式检测和 metadata 一致性弱 | 目标只是通用提取，不强调 Tika 兼容 |
| C：GraalVM/native Tika 包装 | 用 Rust FFI 包一层 native Tika 或现有 Java 逻辑 | 短期功能覆盖最强 | 不是真正 Rust 重写；构建复杂，体积和依赖重，静态库交付困难 | 仅适合过渡验证，不符合当前“以 Rust 重写”目标 |

## 7. 推荐结论

- 推荐方案：选 A；Research 阶段一次性把主题相关的 Tika 能力纳入总体规划，Plan 阶段再按依赖关系、风险和验证成本排序。
- 取舍理由：用户目标是 Rust 静态库/动态库供其他项目使用，不能在调研阶段把主要能力裁掉；同时关键风险仍在 ABI 稳定、解析安全、依赖可控和验证闭环，因此需要先建立调度/结果/检测模型，再按 Plan 顺序推进各功能族。
- 需要进入 Plan 的关键约束：
  - Cargo 输出建议同时包含 `rlib`、`staticlib`、`cdylib`；C ABI 用 `extern "C"`、opaque handle、显式 free 函数，避免跨 ABI 传 Rust `String`/`Vec`/trait object。
  - C ABI 建议返回 UTF-8 JSON 结果，字段包含 `mime_type`、`metadata`、`content`、`embedded`、`warnings`、`error`，保留多值 metadata。
  - 所有 parser 必须有资源限制：输入大小、解压比例、递归深度、嵌入文件数量、输出字符上限、超时/取消、临时文件策略。
  - native 依赖必须 feature-gate；默认包尽量 pure Rust，避免静态库链接被系统库污染。
  - MIME 数据库应从 `tika-mimetypes.xml` 转为构建期生成的 Rust 表或压缩资源，保留 magic/glob/rootXML/subtype/alias 规则。
- 需要用户确认的问题：
  - Plan 阶段的实现顺序：先按基础设施依赖排序，还是先按业务格式价值排序？
  - 目标平台：Linux only，还是 Linux/macOS/Windows；是否需要 musl 静态链接？
  - 是否允许依赖 C/C++/系统库，例如 PDFium、Poppler、Tesseract、libmagic？
  - 对 Tika 兼容性的要求：MIME 类型/metadata key/content 文本是否需要与 Tika 回归对齐？
  - FFI 消费方语言：C/C++、Go、Python、Java/JNI、Node，决定 header、符号命名和内存释放约定。
- 后续验证方向：
  - 从 Tika `tika-core` 和 `tika-parsers` 测试资源抽样建立 golden corpus。
  - 首先验证 MIME detection：扩展名、magic、XML root、ZIP/OLE 容器 specialization。
  - 再验证提取：TXT/CSV/HTML/XML、PDF、Office/OOXML/ODF/EPUB/iWork、旧 Office/OLE、压缩包、邮件、图片音视频、OCR/NLP/专业格式的文本、metadata、嵌入文档和增强结果。
  - 对不可信输入建立 fuzzing 与 resource-limit 测试，尤其是 zip bomb、深递归 XML、损坏 PDF/Office。

## 8. Research 阶段审视

- 安全审查员：已按 L4 处理；本阶段只读 Tika、只新增当前任务文档；后续实现将命中 FFI/ABI、解析不可信输入、构建链接门禁，必须在 Plan 中明确 C ABI、内存释放、panic 边界、资源限制和回滚策略。
- 高级产品：需求边界已修正为“主题相关能力全部纳入总体规划，server/app/batch/eval 等运行形态不纳入库核心”；Research 不再裁掉 OCR/NLP/翻译/深度学习/专业格式等相关能力。
- 高级架构师：推荐先做 core 和 registry，再做 parser 与增强能力插件化集成，避免把 Java 多模块和 ServiceLoader 模型照搬到 Rust；静态/动态库交付会反向约束依赖选择。
- 高级工程师：调研尚未包含运行时基准和 crate POC，不能直接进入编码；下一阶段应先产出完整功能地图、排序依据、最小 API 和 POC 验证计划。

## 9. 参考资料

- 本地源码：`/data/source/tika/pom.xml`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/Tika.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/config/TikaConfig.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/detect/DefaultDetector.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/mime/MimeTypes.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/parser/AutoDetectParser.java`
- 本地源码：`/data/source/tika/tika-core/src/main/java/org/apache/tika/parser/CompositeParser.java`
- 本地源码：`/data/source/tika/tika-core/src/main/resources/org/apache/tika/mime/tika-mimetypes.xml`
- 本地源码：`/data/source/tika/tika-parsers/src/main/resources/META-INF/services/org.apache.tika.parser.Parser`
- 本地源码：`/data/source/tika/tika-parsers/src/main/resources/META-INF/services/org.apache.tika.detect.Detector`
- 本地源码：`/data/source/tika/tika-core/src/main/resources/org/apache/tika/parser/external/tika-external-parsers.xml`
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
