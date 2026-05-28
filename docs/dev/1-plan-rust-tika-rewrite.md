# Rust 重写 Tika 内容与文件类型提取开发计划

> 文档元数据
> - 文件编号：1
> - 文档类型：plan
> - 文件路径：docs/dev/1-plan-rust-tika-rewrite.md
> - 文档版本：v1.0.0
> - 最后更新：2026-05-27
> - 关联需求：以 Rust 重写 `/data/source/tika` 的内容提取、元数据提取和文件类型识别能力，并提供静态库与动态库供其他项目使用。
> - 关联调研：docs/dev/1-research-rust-tika-rewrite.md

## 1. 目标与成功标准

- 任务目标：把 Tika 中与库化内容提取、元数据提取、文件类型识别、嵌入文档递归、内容增强、结果序列化相关的能力一次性纳入 Rust 重写规划；执行顺序只表示依赖和验收节奏，不表示裁掉或放弃功能。
- 成功标准：
  - Rust workspace 可构建 `rlib`、`staticlib`、`cdylib`，并提供稳定 C ABI、头文件、版本查询和显式内存释放函数。
  - 对外能力覆盖 `detect`、`parse`、`parse_file`、配置加载、资源限制、错误/告警返回、metadata 多值输出、嵌入文档递归输出。
  - MIME/file type detection 覆盖 Tika MIME 数据库、magic、glob、rootXML、alias、subclass、override、ZIP/OLE/BPList 容器 specialization 和 encoding detector。
  - Parser 和增强能力覆盖调研文档中列出的全部 Tika 相关功能族与 75 个 parser SPI 类名，并保留 provider 级映射。
  - 所有不可信输入路径具备资源限制：输入大小、读取窗口、解压比例、递归深度、嵌入数量、输出字符数、超时/取消、临时文件清理。
  - 验证体系覆盖 Tika 测试资源 golden corpus、检测差异、格式提取差异、嵌入文档、ABI/linkage、fuzzing、性能和发布包验证。
- 前置条件：
  - 默认先按 Linux/glibc、pure Rust 优先、native 依赖 feature-gate、Tika 行为兼容优先规划。
  - 目标平台、是否要求 musl/Windows/macOS、是否允许 PDFium/Poppler/Tesseract/GDAL 等 native 依赖、FFI 主要消费语言仍需在执行前确认。
- 非目标：
  - 不实现 `tika-server`、`tika-app`、`tika-batch`、REST 服务、批处理产品形态或 Java/GraalVM 包装层。
  - 不修改 `/data/source/tika`；该仓库只作为源码、测试资源和行为规格参考。

## 2. 修改边界

- 最大修改范围：新增/调整 Rust workspace、构建脚本、核心库、parser/增强插件模块、FFI 层、测试资源索引、验证脚本、发布文档和本任务文档。
- 禁止触碰范围：不改 `/data/source/tika`；不提交无关 `.codex`；不做破坏性清理、历史重写、批量格式化或无关重构。
- 影响模块/文件：
  - 规划新增：`Cargo.toml`、`crates/vectraparse-core`、`crates/vectraparse-mime`、`crates/vectraparse-parsers`、`crates/vectraparse-enhance`、`crates/vectraparse-ffi`、`include/`、`tests/fixtures`、`tests/golden`、`benches/`、`fuzz/`、`docs/`。
  - 规划更新：`docs/dev/1-plan-rust-tika-rewrite.md`、后续 Summary、必要时 `docs/overview-product-dev.md`。
- 依赖关系：core/result/resource-limit/metadata 是所有 parser 前置；MIME detector 是 parser 调度前置；容器递归是 Office、压缩包、邮件附件、嵌入文档前置；FFI 和发布包贯穿每个可交付里程碑。

## 3. 安全门禁摘要

| 项 | 结论 |
|----|------|
| 风险矩阵结论 | L4，新系统与长期架构变化，涉及 ABI、构建链接、不可信文件解析、安全与性能 |
| 命令权限 | 文档阶段 C0/C1；后续依赖下载、网络访问、系统库安装、发布推送属于 C2，需确认；破坏性操作属于 C3，默认禁止 |
| 高风险开发门禁 | 是：Rust FFI/ABI、内存释放、并发/取消、解析不可信输入、构建链接、native 依赖 |
| 破坏性操作 | 无计划；如需删除/迁移/重写历史必须单独确认 |
| 用户确认事项 | 目标平台、native 依赖许可、Tika 兼容级别、FFI 消费语言、默认启用功能、模型/外部服务密钥策略 |
| 止损/回滚方案 | 每个里程碑独立提交；公共 API 变更走版本号和 header 兼容检查；native/外部能力全部 feature-gate，可关闭回退到核心提取链路 |

## 4. 总体架构计划

- `vectraparse-core`：输入抽象、资源限制、错误类型、metadata schema、解析结果、parser/detector trait、递归嵌入模型、配置模型、插件注册。
- `vectraparse-mime`：由 Tika MIME XML 生成或加载 Rust 表，提供 magic/glob/rootXML/alias/subclass/specialization、容器 detector 和 encoding detector。
- `vectraparse-parsers`：格式 parser 集合，按功能族分 feature，依赖 core 和 mime，不直接暴露 FFI。
- `vectraparse-enhance`：语言识别、翻译、NLP/NER、情感分析、cTAKES、深度学习识别、OCR/external parser 适配；默认可关闭。
- `vectraparse-ffi`：`staticlib`/`cdylib` 出口、C ABI、opaque handle、JSON 结果、错误码、内存释放、版本和能力查询。
- 测试与验证：`tests/golden` 保存 Tika 对照样本和期望；`fuzz` 覆盖 detector、容器、解析入口；`benches` 覆盖检测和常见格式提取性能。

## 5. 执行计划 / Todo

> 状态说明：本表是完整范围 todo；所有条目均纳入规划。状态只表示当前执行进度，不表示功能是否属于范围外。

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| P0-01 | 确认目标平台、musl/Windows/macOS、native 依赖许可、Tika 兼容级别、FFI 消费语言和默认 feature 策略 | 在 Plan/Summary 记录确认项；未确认项保持 feature-gate 默认关闭 | 完成 |
| P0-02 | 建立 Rust workspace 和 crate 边界：core、mime、parsers、enhance、ffi、测试/bench/fuzz | `cargo metadata`、`cargo check --workspace` | 完成 |
| P0-03 | 配置 `rlib`、`staticlib`、`cdylib` 输出、符号命名、版本脚本或导出控制 | `cargo build --release` 后检查 `.a`/`.so` 和导出符号 | 完成 |
| P0-04 | 设计 C ABI：opaque handle、options、result buffer、error code、free 函数、version/capabilities API | C 示例编译链接并调用 detect/parse/free；ABI header 审查 | 完成 |
| P0-05 | 定义配置模型：启用 parser、资源限制、外部命令、模型路径、服务密钥引用、metadata 输出选项 | 配置反序列化单测；非法配置错误路径单测 | 完成 |
| P0-06 | 定义错误、告警和资源限制模型，保证 panic 不跨 FFI 边界 | 单测覆盖错误分类、panic catch、资源释放 | 完成 |
| P0-07 | 定义 metadata schema 与多值存储，映射 Tika `AccessPermissions`、`Office`、`PDF`、`XMP*` 等命名空间 | schema 对照测试；JSON round-trip 测试 | 完成 |
| P0-08 | 建立 parser/detector registry，替代 Java ServiceLoader，支持排序、feature 裁剪和能力查询 | registry 单测；provider 列表快照测试 | 完成 |
| P0-09 | 建立 golden corpus 目录、样本 manifest、Tika oracle 输出格式和差异比较器 | 样本索引校验；空 corpus 和最小样本对照通过 | 完成 |
| P0-10 | 引入 CI/本地验证入口：fmt、clippy、test、ABI smoke、golden、fuzz smoke、bench smoke | 文档化验证命令；本地最小流水线通过 | 完成 |
| P1-01 | 转换 `tika-mimetypes.xml`，生成 mime-type、alias、subclass、magic、glob、rootXML 表 | 与 Tika XML 计数对齐；生成表快照测试 | 完成 |
| P1-02 | 实现 `MediaTypeRegistry` 等价能力：normalize、alias、specialize、supertype 查询 | Tika registry 用例对照测试 | 完成 |
| P1-03 | 实现 magic matcher：读取窗口、优先级、offset、mask、空文件识别 | magic golden 测试；损坏/短输入测试 | 完成 |
| P1-04 | 实现 glob/name/hint/content-type override 检测链路 | 扩展名、资源名、显式 content-type 对照测试 | 完成 |
| P1-05 | 实现 XML/HTML/root element 细分检测和文本/二进制回退 | XML root、HTML meta charset、plain text 样本测试 | 完成 |
| P1-06 | 实现 ZIP 容器 detector：OOXML、ODF、EPUB、iWork、JAR/APK 等 specialization | ZIP 内部路径 golden 测试；zip bomb 限制测试 | 完成 |
| P1-07 | 实现 OLE/POIFS detector 和 CFB 基础读取 | DOC/XLS/PPT/MSG/Access/owner file 检测样本对照 | 完成 |
| P1-08 | 实现 BPList detector 和 Apple plist 识别 | binary plist/XML plist 样本对照 | 完成 |
| P1-09 | 实现高级 detector：`OverrideDetector`、`ZeroSizeFileDetector`、`NameDetector`、`TypeDetector`、`TrainedModelDetector`、`NNExampleModelDetector` | provider 映射测试；模型 detector 可关闭/可配置测试 | 完成 |
| P1-10 | 实现 encoding detector：HTML、universal、ICU4J 等价或替代策略 | 多编码文本、HTML charset、BOM、错误编码对照测试 | 完成 |
| P1-11 | 暴露 detect API：内存、文件路径、带 hint 输入、能力查询 | Rust API 和 C ABI detect smoke/golden 测试 | 完成 |
| P2-01 | 实现输入抽象：内存、文件、流、mark/reset、临时资源、读取上限 | 大文件、短读、临时文件清理、并发读取测试 | 完成 |
| P2-02 | 实现 parser trait、composite parser、fallback、supplementing、multiple parser 调度 | MIME 到 parser 映射测试；fallback 路径测试 | 完成 |
| P2-03 | 实现结构化结果：content、metadata、embedded、warnings、errors、timing、parser chain | JSON schema 和 round-trip；嵌入路径快照测试 | 完成 |
| P2-04 | 实现内容处理器：text/html/xml 输出、XPath 过滤、link/phone/standard 抽取、XMP handler | handler 单测；输出字符上限测试 | 完成 |
| P2-05 | 实现递归嵌入文档处理：深度、数量、路径、父子 metadata、异常隔离 | 嵌入 Office/zip/mail 样本；递归限制测试 | 完成 |
| P2-06 | 实现 `ContainerExtractor`、`ParserContainerExtractor`、`Embedder`、`ExternalEmbedder` 等等价边界 | API 单测；禁用 embedder 时行为测试 | 完成 |
| P2-07 | 实现 digest/hash parser、fork 隔离等价策略、NetworkParser 边界和可关闭策略 | hash 输出测试；隔离/禁用策略审查 | 完成 |
| P3-01 | 实现 TXT parser 和编码归一化 | TXT 多编码 golden；空文件和二进制误判测试 | 完成 |
| P3-02 | 实现 CSV/TSV/分隔文本 parser，覆盖 `TextAndCSVParser` | CSV 方言、转义、坏行、metadata 测试 | 完成 |
| P3-03 | 实现 HTML parser，提取正文、title、meta、链接和 charset | HTML golden；恶意/深层 DOM 资源限制测试 | 完成 |
| P3-04 | 实现 XML parser、DcXML、FictionBook、通用 XML metadata 和 XPath | XML root golden；XXE 禁止测试 | 完成 |
| P3-05 | 实现 source code parser 和语言识别/metadata | 多语言源码样本；纯文本兜底差异测试 | 待开始 |
| P3-06 | 实现 strings/Latin1 strings parser | 二进制 strings 样本；输出上限测试 | 待开始 |
| P3-07 | 实现 feed parser：RSS、Atom 等 | feed golden；坏 XML 降级测试 | 待开始 |
| P3-08 | 实现 IPTC ANPA、XLIFF/XLZ、DIF、ENVI header 等文本衍生格式 | 每格式最小 golden；MIME 检测联动测试 | 待开始 |
| P3-09 | 实现 AppleSingle、PList、FictionBook、DcXML 等轻量专用 parser | SPI 类名映射快照；格式样本测试 | 待开始 |
| P4-01 | 实现压缩/包 parser：Package、Compressor、RAR，以及 zip/tar/gzip/bzip2/xz/zstd/7z 策略 | 解压 golden；解压比例/深度/数量限制测试 | 待开始 |
| P4-02 | 实现 OOXML：docx/xlsx/pptx、关系文件、core props、嵌入文件、旧 Excel XML/WordML/SpreadsheetML | Office golden；嵌入附件和公式/表格样本测试 | 待开始 |
| P4-03 | 实现 ODF/OpenDocument/OpenOffice parser | odt/ods/odp golden；manifest 和 metadata 测试 | 待开始 |
| P4-04 | 实现 EPUB parser | EPUB spine/metadata/嵌入资源 golden | 待开始 |
| P4-05 | 实现 iWork package parser | pages/numbers/keynote 样本对照 | 待开始 |
| P4-06 | 实现 PDF parser：文本、metadata、附件、权限、加密、OCR hook、preflight | PDF golden；加密/损坏/大文件/附件测试 | 待开始 |
| P4-07 | 实现 OLE/CFB 旧 Office：DOC/XLS/PPT、OfficeParser、OldExcel、MSOwnerFile | OLE golden；宏/嵌入对象安全审查 | 待开始 |
| P4-08 | 实现 Microsoft 专有格式：OneNote、Access/Jackcess、TNEF、EMF、WMF、MSG/PST 联动 | 样本 golden；native/feature 依赖检查 | 待开始 |
| P4-09 | 实现 RTF 和 RTF object data 递归 | RTF golden；嵌入对象测试 | 待开始 |
| P4-10 | 实现 HWP、CHM、WordPerfect、Quattro Pro 等其他文档格式 | 每格式 golden；检测与 parser 映射测试 | 待开始 |
| P5-01 | 实现 RFC822/MIME 邮件 parser、附件递归和 charset 处理 | eml golden；多附件和坏 charset 测试 | 待开始 |
| P5-02 | 实现 MBOX parser | mbox 多邮件样本；递归附件测试 | 待开始 |
| P5-03 | 实现 Outlook PST/MSG/TNEF 邮箱能力 | pst/msg/tnef golden；大邮箱资源限制测试 | 待开始 |
| P5-04 | 实现图片 metadata：通用 image、JPEG、TIFF、BPG、PSD、WebP、HEIF、ICNS、EXIF/XMP/IPTC | 图片格式 golden；损坏图片 fuzz smoke | 待开始 |
| P5-05 | 实现音频 metadata：Audio、MP3、MIDI | mp3/id3/midi golden；坏 tag 测试 | 待开始 |
| P5-06 | 实现视频 metadata：MP4、FLV 和通用 video | mp4/flv golden；大文件读取窗口测试 | 待开始 |
| P5-07 | 实现 captioning 和对象识别接入点 | 模型/服务禁用、超时、失败降级测试 | 待开始 |
| P6-01 | 实现数据库/表格：DBF、SQLite、Access、JDBC 等价边界 | dbf/sqlite/access golden；连接型能力默认禁用审查 | 待开始 |
| P6-02 | 实现科学数据：NetCDF、HDF、GRIB、MAT、SAS | 每格式 metadata/text golden；native 依赖 feature 测试 | 待开始 |
| P6-03 | 实现地理/工程数据：GDAL、DWG、Geo、GeographicInformation | 样本 golden；GDAL/native 依赖隔离测试 | 待开始 |
| P6-04 | 实现 ISA-Tab、Grobid/Journal、Pooled Time Series、POT、PRT 等专业格式 | 每格式 golden；外部服务/模型降级测试 | 待开始 |
| P6-05 | 实现 crypto/security 格式：PKCS#7、TSD、密码 provider、加密文档错误模型 | 加密样本、密码正确/错误、权限 metadata 测试 | 待开始 |
| P6-06 | 实现 Java class、可执行文件、AFM/TrueType 字体 parser | class/ELF/PE/Mach-O/font golden；安全扫描限制测试 | 待开始 |
| P7-01 | 实现语言识别 API、n-gram 资源和 `LanguageIdentifier` 等价能力 | 多语种 corpus；低置信度/短文本测试 | 待开始 |
| P7-02 | 实现语言 provider：Optimaize、Text、Lingo24 等价或替代 | provider 切换、禁用、配置错误测试 | 待开始 |
| P7-03 | 实现翻译 provider：Microsoft、Google、Lingo24、Cached、Joshua、Moses、Yandex 的配置边界 | mock 服务、缓存、密钥缺失、超时降级测试 | 待开始 |
| P7-04 | 实现 NLP/NER 多后端和 sentiment | mock/model 样本；模型缺失降级测试 | 待开始 |
| P7-05 | 实现 cTAKES 医学文本集成边界 | 医学样本 golden；外部依赖关闭测试 | 待开始 |
| P7-06 | 实现深度学习 recognition 和 captioning 模型接入 | 模型路径、批量资源限制、失败降级测试 | 待开始 |
| P7-07 | 实现 OCR/Tesseract 和 external parser XML 配置 | OCR 样本、命令超时、沙箱/路径限制测试 | 待开始 |
| P8-01 | 实现 XMP 映射和 metadata 标准化输出 | XMP golden；多值 metadata round-trip | 待开始 |
| P8-02 | 实现 JSON 序列化、稳定字段、schema version 和兼容策略 | JSON schema 测试；旧字段兼容测试 | 待开始 |
| P8-03 | 实现 FFI 包装完整链路：detect/parse/options/result/free/capabilities | C 集成测试；重复 free/空指针/错误路径测试 | 待开始 |
| P8-04 | 实现发布包：头文件、pkg-config/cmake 文件、license manifest、feature 矩阵 | 本地 install/link smoke；license 审查 | 待开始 |
| P8-05 | 编写使用文档：Rust API、C ABI、错误码、资源限制、feature、样例 | 文档示例编译；README 审查 | 待开始 |
| P9-01 | 建立 Tika 对照检测 golden：magic/glob/rootXML/container/encoding | 差异报告；必须解释所有偏差 | 待开始 |
| P9-02 | 建立格式提取 golden：content、metadata、embedded、warnings、errors | 按功能族生成差异报告和通过门槛 | 待开始 |
| P9-03 | 建立安全测试：zip bomb、深递归 XML、损坏 PDF/Office、超大 metadata、路径穿越 | 所有资源限制命中后状态一致且资源释放 | 待开始 |
| P9-04 | 建立 fuzzing：detector、MIME XML 表、ZIP/OLE、PDF/Office 入口、FFI JSON | fuzz smoke 和定期长跑报告 | 待开始 |
| P9-05 | 建立性能基准：检测吞吐、常见格式解析耗时、峰值内存、并发解析 | bench 基线；性能回归阈值 | 待开始 |
| P9-06 | 建立 ABI/linkage 验证：staticlib/cdylib、C/C++、Go/Python/Java/JNI 可选消费方 | 多语言 smoke；符号和内存释放检查 | 待开始 |
| P9-07 | 建立发布验收：clean checkout、无网络/有缓存构建、feature 组合、包体积、许可证 | release checklist 全通过 | 待开始 |

## 6. 里程碑验收

| 里程碑 | 验收范围 | 出口标准 | 状态 |
|--------|----------|----------|------|
| M0 基础架构 | workspace、core、FFI、配置、资源限制、测试框架 | `detect`/空 parser 可通过 Rust 和 C ABI 调用，错误路径可释放资源 | 完成 |
| M1 检测闭环 | MIME 数据库、detector、encoding、容器 specialization | Tika 检测 golden 通过，ZIP/OLE/BPList 不被粗略误判 | 完成 |
| M2 基础提取 | TXT/CSV/HTML/XML/source/strings/feed/轻量文本格式 | 结果 JSON、metadata、资源限制、差异报告稳定 | 待开始 |
| M3 文档与容器主链路 | 压缩、OOXML、ODF、EPUB、iWork、PDF、旧 Office/OLE、RTF、HWP/CHM/WordPerfect | 常见文档主链路可递归提取正文、metadata 和嵌入文档 | 待开始 |
| M4 邮件与媒体 | RFC822/MBOX/PST/MSG/TNEF、图片、音频、视频、captioning/recognition 接入 | 邮件附件和媒体 metadata 通过 golden，外部能力可关闭 | 待开始 |
| M5 专业格式与安全格式 | 数据库、科学数据、地理工程、crypto、Java/可执行、字体 | 每个专业格式有样本和 feature 策略，native 依赖不污染默认包 | 待开始 |
| M6 内容增强 | 语言识别、翻译、NLP/NER、sentiment、cTAKES、OCR/external parser | 模型/服务能力可配置、可降级、可测试 | 待开始 |
| M7 发布硬化 | ABI、fuzz、性能、文档、license、release 包 | 静态库/动态库可交付给外部项目使用 | 待开始 |

## 7. 验证计划

- 基础验证：`cargo fmt --check`、`cargo clippy --workspace --all-features`、`cargo test --workspace`、`cargo build --release`。
- ABI/API 验证：编译 C 示例，分别链接 `staticlib` 和 `cdylib`；检查 null pointer、重复释放、错误码、panic 边界和跨线程调用策略。
- 行为验证：从 `/data/source/tika` 测试资源建立 golden corpus，对 detect、parse、metadata、embedded、warnings、errors 做差异报告。
- 安全验证：fuzz detector/parser 入口；覆盖压缩炸弹、递归炸弹、损坏文档、超大文件、路径穿越、外部命令超时。
- 性能验证：检测吞吐、常见文档解析耗时、峰值内存、并发解析和输出字符上限。
- 不可执行验证项：需要真实外部服务或模型的翻译、NER、cTAKES、DL、OCR，可先用 mock 和禁用路径验证，真实环境验收单独记录。
- 残余风险：Tika 行为庞大且依赖格式细节，计划完成前仍需逐格式 golden 和真实样本补证据；native 依赖选择会影响静态链接、发布体积和跨平台可用性。

## 8. Plan 阶段审视

- 安全审查员：计划已把 FFI/ABI、资源释放、panic 边界、外部命令、native 依赖和不可信输入资源限制列为前置任务；后续进入 Code 前必须逐里程碑更新状态和验证结果。
- 高级产品：用户要求的 Tika 相关能力已全部纳入 todo，执行顺序只表达依赖关系；server/app/batch 等运行形态仍不进入库核心。
- 高级架构师：模块边界按 core/mime/parsers/enhance/ffi 分离，避免 parser 直接绑定 ABI；native 和服务能力使用 feature-gate 降低默认交付风险。
- 高级工程师：todo 均附验证方式；风险最高的 detector、资源限制、FFI 和 golden corpus 排在前面，避免先堆 parser 后无法验证行为。

## 9. 变更记录

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-05-27 | 创建完整 Plan 和 todo 清单 | 用户要求把调研结论整合为完整计划并列出详细 todo |
| 2026-05-27 | 完成 P0-01 约束确认 | 按默认策略固定第一阶段执行边界，避免后续任务阻塞 |
| 2026-05-27 | 完成 P0-02 workspace 骨架 | 建立多 crate 工作区和依赖边界，进入可编译状态 |
| 2026-05-27 | 完成 P0-03 构建产物配置 | `vectraparse-ffi` 输出 `rlib/staticlib/cdylib` 并通过 release 构建验收 |
| 2026-05-27 | 完成 P0-04 C ABI 初版 | 增加 opaque handle/options/result/error API、头文件与 C smoke 调用样例 |
| 2026-05-27 | 完成 P0-05 配置模型初版 | 在 core 中引入 KV 反序列化配置和非法输入错误路径单测 |
| 2026-05-27 | 完成 P0-06 错误与资源限制模型 | 增加 limits/failure/warning 模型并在 FFI 边界加入 panic 捕获 |
| 2026-05-27 | 完成 P0-07 metadata schema 初版 | 引入多值 metadata 存储和 JSON round-trip，覆盖 AccessPermissions/Office/PDF/XMP 命名空间 |
| 2026-05-27 | 完成 P0-08 provider registry 初版 | 建立 detector/parser 注册、排序、feature 标记和能力查询机制 |
| 2026-05-27 | 完成 P0-09 golden corpus 骨架 | 建立 manifest、期望结果文件和最小差异比较脚本 |
| 2026-05-27 | 完成 P0-10 统一验证入口 | 增加 Makefile 验证目标和 README 命令文档，形成最小本地流水线 |
| 2026-05-27 | 完成里程碑 M0 基础架构 | P0-01 到 P0-10 全部完成，`make pipeline` 通过 |
| 2026-05-27 | 完成 P1-01 MIME 计数生成 | 引入 build.rs 读取 `tika-mimetypes.xml` 并生成计数常量及对齐测试 |
| 2026-05-28 | 完成 P1-02 `MediaTypeRegistry` 初版 | 在 `vectraparse-mime` 新增 normalize/alias/specialize/supertype 能力，并通过定向单元测试 |
| 2026-05-28 | 完成 P1-03 magic matcher 初版 | 在 `vectraparse-mime` 新增读取窗口、优先级、offset、mask、空文件识别与短输入回退测试 |
| 2026-05-28 | 完成 P1-04 检测优先链路初版 | 在 `vectraparse-mime` 新增 force/hint/resource name/glob/magic 优先链路与回退测试 |
| 2026-05-28 | 完成 P1-05 XML/HTML/root 与文本回退初版 | 在 `vectraparse-mime` 增加 XML root 细分、HTML 检测和 plain text 回退测试 |
| 2026-05-28 | 完成 P1-06 ZIP 容器细分初版 | 在 `vectraparse-mime` 增加 OOXML/ODF/EPUB/iWork/JAR/APK 的 ZIP specialization 检测 |
| 2026-05-28 | 完成 P1-07 OLE/POIFS 细分初版 | 在 `vectraparse-mime` 增加 CFB magic 与 DOC/XLS/PPT/MSG/Access 基础识别 |
| 2026-05-28 | 完成 P1-08 plist 检测初版 | 在 `vectraparse-mime` 增加 binary plist magic 和 XML plist 特征识别 |
| 2026-05-28 | 完成 P1-09 高级 detector 配置初版 | 在 `vectraparse-mime` 增加 detector 开关、provider 映射与模型 detector 可配置测试 |
| 2026-05-28 | 完成 P1-10 编码检测初版 | 在 `vectraparse-mime` 增加 BOM、HTML charset 与 UTF-8/binary 回退检测 |
| 2026-05-28 | 完成 P1-11 detect API 对外暴露 | 在 Rust/C ABI 提供内存、文件路径、hints 检测接口并更新 capabilities |
| 2026-05-28 | 完成里程碑 M1 检测闭环 | P1-01 到 P1-11 已完成并通过本地定向测试 |
| 2026-05-28 | 完成 P2-01 输入抽象初版 | 在 `vectraparse-core` 增加 Memory/File 输入源、读取窗口和读取上限校验 |
| 2026-05-28 | 完成 P2-02 parser 调度初版 | 在 `vectraparse-parsers` 增加 trait/composite/fallback/supplementing/multiple 调度与测试 |
| 2026-05-28 | 完成 P2-03 结构化结果初版 | 在 `vectraparse-core` 增加 StructuredResult 模型、JSON round-trip 与 parse/detect 结构化输出 |
| 2026-05-28 | 完成 P2-04 内容处理器初版 | 在 `vectraparse-parsers` 增加 text/html/xml handler、链接/电话抽取与字符上限截断测试 |
| 2026-05-28 | 完成 P2-05 递归嵌入处理初版 | 在 `vectraparse-parsers` 增加嵌入递归、路径追踪、深度/数量限制和异常隔离告警 |
| 2026-05-28 | 完成 P2-06 容器提取边界初版 | 在 `vectraparse-parsers` 增加 ContainerExtractor/Embedder 边界与禁用 embedder 行为测试 |
| 2026-05-28 | 完成 P2-07 安全与摘要 parser 初版 | 在 `vectraparse-parsers` 增加 digest/hash parser、network parser 禁用策略与隔离配置测试 |
| 2026-05-28 | 完成 P3-01 TXT parser 初版 | 在 `vectraparse-parsers` 增加编码归一化解码、空文件处理和二进制误判规避测试 |
| 2026-05-28 | 完成 P3-02 CSV/TSV parser 初版 | 在 `vectraparse-parsers` 增加方言检测、转义处理、坏行检测和结构化 metadata 输出 |
| 2026-05-28 | 完成 P3-03 HTML parser 初版 | 在 `vectraparse-parsers` 增加正文/title/meta/link/charset 提取与深层输入告警测试 |
| 2026-05-28 | 完成 P3-04 XML parser 初版 | 在 `vectraparse-parsers` 增加 XML root/profile 提取与 XXE 阻断测试 |

## 10. P0-01 默认确认结果

- 目标平台：Linux x86_64 + glibc（首发），macOS/Windows 和 musl 作为后续扩展目标，当前保持禁用状态。
- native 依赖策略：默认 pure Rust；PDFium/Poppler/Tesseract/GDAL 等通过 feature-gate 控制，默认关闭。
- 兼容策略：MIME 检测与 metadata key 命名优先对齐 Tika；内容文本允许在 whitespace 层面存在可解释差异。
- FFI 首要消费语言：C（后续再扩展 C++/Go/Python/JNI）。
- 默认 feature：仅 core/mime/基础 parser；增强能力（翻译、NLP、DL、OCR、外部命令）默认关闭。
