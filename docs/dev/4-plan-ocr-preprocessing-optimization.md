# OCR 预处理与入口识别优化开发计划

> 文档元数据
> - 文件编号：4
> - 文档类型：plan
> - 文件路径：docs/dev/4-plan-ocr-preprocessing-optimization.md
> - 文档版本：v1.0.6
> - 最后更新：2026-06-01
> - 关联需求：按当前代码实现校准 OCR 预处理优化计划与完成状态；按识别质量优化建议实现一版可测试改进；继续实现增强/上采样/旋转重跑 det、部分成功补跑合并、det NMS/unclip 扩边、透明图黑底增强、复杂截图区域级布局聚类、颜色背景区域 fallback、结构化 regions/lines 输出和 trace metadata。
> - 关联调研：当前为代码只读分析结论，未单独创建 research 文档。

## 1. 目标与成功标准

- 任务目标：
  - 提升图片 OCR 入口命中率，避免 JPEG/TIFF/WebP/BMP/GIF 等图片因 MIME 识别不足而绕过 OCR。
  - 提升 OCR 对低分辨率、小字、低对比、白字黑底、长行、英文行、倾斜/旋转图片的文本提取成功率。
  - 增强 OCR 失败路径可观测性，能区分未进入 OCR、解码失败、检测无框、识别为空和低置信度输出。
- 成功标准：
  - 常见图片格式能稳定进入 `ImageMetadataParser` 的 OCR 路径。
  - 失败样本可按原因分桶，并在 metadata 或 warnings 中体现。
  - 当前单元测试覆盖 MIME 入口、诊断 metadata/warnings、图像增强变体、小图上采样、det mask 后处理、中英文结果选择、旋转变体、行级质量过滤、区域级布局聚类、颜色背景区域候选、结构化区域/行输出和 OLE 图片候选处理。
  - 当前 golden 仅覆盖 `image/png` parser 叶子行为和 `image.format` 基础元数据；尚未落地真实 OCR 文本 fixture。
  - 当前验证主要证明入口、诊断和辅助算法行为；尚未用真实失败样本系统证明 OCR 文本质量提升。
- 前置条件：
  - 收集 5-10 张当前无法提取文字的真实失败样本，或构造等价最小样本。
  - 明确是否允许新增纯 Rust 图像处理依赖；默认优先使用现有 `image` crate 能力。
- 非目标：
  - 不更换 OCR 模型，不引入外部 OCR 服务或系统二进制。
  - 不重写 ONNX Runtime FFI 层。
  - 不在本任务中实现 PDF 渲染后 OCR；PDF OCR 仅保持现有 hook/告警边界。

## 2. 修改边界

- 最大修改范围：
  - `crates/vectraparse-mime/src/lib.rs`
  - `crates/vectraparse-parsers/src/lib.rs`
  - `crates/vectraparse-ocr/src/lib.rs`
  - `crates/vectraparse-mso-binary/src/lib.rs` 中嵌入图片 OCR 调用和候选过滤逻辑
  - 相关 tests/fixtures、golden manifest、开发文档
- 禁止触碰范围：
  - 不修改 `crates/vectraparse-ocr/src/ort.rs` 的 unsafe FFI 调用，除非后续发现明确 ORT 输入/输出根因并重新评估风险。
  - 不调整构建链接、ONNX 模型文件和字典文件内容。
  - 不引入网络下载、系统包安装或外部命令依赖。
- 影响模块/文件：
  - MIME 检测：补齐图片 magic 和资源名扩展识别。
  - Parser 入口：对齐 `ImageMetadataParser` 支持格式与 OCR 实际解码能力。
  - OCR 核心：当前实现包含原图 det+逐框 rec、区域级布局聚类、颜色背景区域候选 fallback、结构化 `regions/lines` 输出、trace 统计、整图 fallback、luma/HSL/max-channel 的对比度拉伸、Otsu 全局二值化、局部均值二值化、透明图黑底增强变体、小图 1.5x/2x 上采样 fallback、90/180/270 度旋转 fallback、增强/上采样/旋转图重新 det+逐框 rec、行切分 fallback、中英文 rec 结果选择、行级质量过滤、部分成功质量判定和诊断。
  - OCR 核心未实现：长 crop 分段、小角度 deskew、锐化/去噪、真实 OCR 文本 golden；det 后处理仍是轴对齐框，未实现真正 PaddleOCR polygon unclip、旋转框裁剪或语义级页面版面理解。
  - OLE 嵌入图：当前仅 `.doc` 路径汇总图片候选并 OCR；支持候选过滤、去重、预算告警和失败诊断。WebP 支持直接图片流，尚未从复合 payload 中切片提取 WebP。
- 依赖关系：
  - P0 入口与诊断先行；否则无法判断后续预处理是否真正生效。
  - P1 预处理增强依赖可观测指标和失败样本分桶。
  - P2 倾斜/旋转等重处理依赖 P1 结果决定是否必要。

## 3. 安全门禁摘要

| 项 | 结论 |
|----|------|
| 风险矩阵结论 | L3：跨 `mime` / `parsers` / `ocr` / `mso-binary`，影响 OCR 关键链路与输出行为 |
| 命令权限 | C0/C1：只读分析、工作区内文档/代码编辑、定向测试；不需要 C2/C3 |
| 高风险开发门禁 | 是：后续若触碰 OCR FFI、内存布局、构建链接或模型输入输出解释，需重新评估；当前计划默认避开 `ort.rs` |
| 破坏性操作 | 否 |
| 用户确认事项 | 如需新增依赖、替换模型、提交/推送或纳入真实私有样本，需单独确认 |
| 止损/回滚方案 | 每个阶段独立提交小改；若 OCR 质量回退，先禁用新 fallback 或恢复默认预处理路径 |

## 4. Bug 修复计划

- 复现方式：
  - 使用当前失败图片通过 FFI/示例工具解析，记录 MIME、parser_chain、warnings、metadata、content。
  - 将失败样本分为：未进入 OCR、图片解码失败、det 无框、rec 空结果、输出乱码/低置信度。
- 证据等级：E1。已有现象描述和代码根因线索，但尚缺稳定失败样本与自动化复现。
- 当前代码事实与残余短板：
  - 非 PNG 图片入口已补齐到 JPEG/TIFF/WebP/BMP/GIF magic 和常见扩展名回退。
  - 预处理 fallback 已不再只是灰度缩放和平均亮度反色；当前包含 luma/HSL/max-channel 三类灰度源、对比度拉伸、Otsu 全局二值化、反色二值化、局部均值二值化，并在透明图上额外生成黑底增强变体；仍未实现锐化或去噪。
  - det 后处理已从原始连通域扩展为 1 像素 dilation、component score 过滤、轻量 unclip 扩边和 NMS；仍缺少旋转框和更接近 PaddleOCR 的多边形后处理。
  - 中英文 rec 已在逐 crop 和各类 fallback 中统一比较；当前已将动态 rec 宽度上限提升到 `MAX_REC_IMG_W=960`，并保留固定宽度失败回退，避免模型不接受宽输入时整条识别失败。
  - fallback 已从“仅空结果触发”扩展为“空结果、低置信、det 框多但识别行少、有效字符过少或可读比例偏低”触发；补跑候选会按行去重后合并，或在明显更长/更高置信时替换原结果。
  - 检测框识别结果已不再只按全局 y/x 排序；当前会按 bbox 水平重叠、纵向距离、宽度差异将行聚类为区域，区域之间用空行分隔，并在 metadata 中记录 `image.ocr.region_count` 和 `image.ocr.layout_applied`。
  - 颜色背景区域 fallback 会在低质量路径中按粗量化颜色连通域提取明显色块，跳过整页背景和低填充率噪声区域；候选区域内会根据边框背景色做前景色差二值化，再执行 rec，并在 metadata 中记录 `image.ocr.color_region_count`。
  - `OcrResult` 已输出结构化 `regions: Vec<OcrTextRegion>`，每个 region 包含 bbox、文本、置信度、来源和行级 `OcrTextLine`；trace 当前记录最终选中来源、det pass 次数和 fallback 尝试次数，并通过 metadata 写出。
  - 当前新增行级质量过滤会拒绝低置信、重复字符占比过高和标点占比过高的候选，可能降低垃圾输出，也可能丢弃少量真实低置信文本，需要真实样本评估。
- 最小修复点：
  - 先补入口识别和诊断，再引入可回退的预处理增强，不一次性重写 OCR 管线。
- 回归/相关验证：
  - 每个阶段新增或复用失败样本作为定向测试。
  - 保留现有 PNG/OLE 图片 OCR 行为，避免优化导致已可识别样本变空。

## 5. 执行计划

| 步骤 | 修改内容 | 验证方式 | 状态 |
|------|----------|----------|------|
| 1 | 建立 OCR 失败样本分桶与最小复现记录，输出当前 MIME、parser_chain、warnings、metadata、content | 对失败图片运行现有 FFI/示例解析；保存人工记录或测试 fixture | 完成 |
| 2 | 补齐图片入口识别：JPEG/TIFF/WebP/BMP/GIF magic，资源名扩展映射，并对齐 parser 支持格式与 `image` 解码能力 | `cargo test -p vectraparse-mime`; parser 定向测试验证各格式进入图片 parser | 完成 |
| 3 | 增强 OCR 诊断：记录 OCR 是否启用、解码失败、det 框数、rec 空结果、fallback 命中、平均置信度和低置信度告警 | `cargo test -p vectraparse-parsers`; 用失败样本确认 warnings/metadata 能分桶 | 完成 |
| 4 | 增加轻量图像增强 fallback：luma/HSL/max-channel 的对比度拉伸、Otsu 全局二值化、反色二值化、局部均值二值化和透明图黑底增强变体 | `cargo test -p vectraparse-ocr`; 覆盖增强变体、对比度拉伸、Otsu、局部二值化和 alpha 黑底辅助函数 | 完成 |
| 5 | 增加多尺度与小字策略：对小图执行 1.5x/2x 上采样，并在 fallback 中先重新 det+逐框 rec，再保留整图识别兜底，限制最大像素 | 定向 OCR 辅助函数测试；当前未覆盖真实小字 OCR 文本 fixture | 完成 |
| 6 | 改进文本框后处理：det map 做 dilation/box score 过滤/轻量 unclip 扩边/NMS，减少碎框、重复框和裁剪不完整；保留旧逻辑 fallback | OCR det 单元测试或 fixture 对比；验证多行截图顺序稳定 | 完成 |
| 7 | 处理长行与英文行：每个 crop 可按置信度/字符集选择中文或英文模型；动态 rec 宽度上限提升到 `960` 并保留固定宽度失败回退；长 crop 分段未实现 | 当前测试覆盖中英文结果选择和 960 宽度行为；长英文真实 OCR fixture 未落地 | 部分完成 |
| 8 | 增加旋转保守兜底：支持 90/180/270 度旋转图重新 det+逐框 rec，并保留整图识别重试；未实现小角度 deskew | 当前测试覆盖旋转变体生成；未覆盖旋转真实 OCR 文本 fixture | 完成 |
| 9 | 增强 OLE 嵌入图片 OCR 候选处理：仅 `.doc` 路径做图片候选汇总、格式过滤、去重 key 和预算命中诊断，嵌入图失败不影响主文本 | `cargo test -p vectraparse-mso-binary`; 当前 WebP 支持直接流，payload 切片未实现 | 完成 |
| 10 | 收口 golden/文档：当前仅将 `image/png` stub 纳入 parser 提取矩阵；真实 OCR 文本样本尚未进入 golden | 定向替代命令；更新 `docs/dev/4-plan...` 后续状态/总结 | 部分完成 |
| 11 | 增加区域级布局聚类：逐框识别后按 bbox 聚类为区域，区域内按行排序，区域间用空行分隔，并输出区域诊断 metadata | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 12 | 增加颜色背景区域 fallback：按近似背景色连通域提取 UI 色块，在候选区域内做背景色差二值化后识别，并输出颜色区域候选数 metadata | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 13 | 增加结构化区域/行输出和 trace metadata：暴露 OCR region/line bbox、source、confidence，并记录选中来源、det pass 和 fallback 尝试数 | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |

## 6. 验证计划

- 基础验证：
  - `cargo test -p vectraparse-mime`
  - `cargo test -p vectraparse-parsers`
  - `cargo test -p vectraparse-ocr`
  - `cargo test -p vectraparse-mso-binary`
- 高风险验证：
  - 内存：限制多尺度重试最大像素、最大候选数、最大 OCR crop 数。
  - 错误路径：图片解码失败、模型不可用、det/rec 输出异常时必须降级为 warning，不 panic。
  - ABI/API：不修改 C ABI；仅增加 metadata/warnings 字段值时需确认消费者兼容。
  - 性能：对大图和 OLE 多图片样本记录 OCR 预算命中，避免多尺度导致耗时倍增失控。
- 验证环境：
  - 本地 Rust workspace，使用内嵌 `data/det.onnx`、`data/chinese/rec.onnx`、`data/english/rec.onnx` 和字典。
- 不可执行验证项：
  - 没有真实失败样本前，只能用构造样本和辅助函数测试覆盖典型路径，不能证明生产 OCR 文本质量已系统提升。
  - 当前没有真实 OCR 文本 golden，无法自动发现长行、小字、旋转、低对比场景的识别质量回退。
- 残余风险：
  - 预处理增强和变体重跑 det 可能提升召回但带来误识别、重复文本和耗时增加；当前已增加低置信、重复字符、高标点占比过滤、NMS 和行级去重合并，但仍缺少真实样本阈值校准。
  - 行级质量过滤可能减少垃圾输出，也可能丢弃少量真实低置信文本；需要用户测试样本评估阈值是否过严。
  - 部分成功 fallback 的替换/合并策略仍是启发式：可能保留少量原始漏识别片段，也可能在候选更长时替换掉原有短文本。
  - 区域级布局聚类仍是几何启发式，不理解页面语义；复杂表格、嵌套卡片、瀑布流、多弹窗叠加或跨区域标题仍可能排序不符合人工阅读顺序。
  - 颜色背景区域 fallback 使用粗颜色量化和边框背景估计；渐变、阴影、图片背景、透明叠层或彩色插图可能产生候选过多或候选缺失，当前只在低质量 fallback 路径触发并限制最大候选数。
  - 结构化 bbox 对原图 det、增强图 det、上采样和旋转 det 做了坐标回映；复杂旋转/裁剪组合仍需真实截图验证坐标是否足够准确。
  - 不更换模型的前提下，极端模糊、严重透视、手写字或超低分辨率图片仍可能无法可靠提取。

当前验证记录：
- 已固化最小复现桶 1：JPEG 字节在当前 FFI `parse_json_runtime` 路径中被识别为 `application/octet-stream`，`parser_chain=["StringsParser"]`，证明“未进入 OCR”现象可稳定复现。
- 已固化最小复现桶 2：损坏图片在 `ImageMetadataParser` 路径中返回 `image-corrupted-or-unknown`，并伴随 `image-ocr-failed` 或 `image-ocr-model-unavailable`，证明“图片解码失败/无法进入识别”现象可稳定复现。
- 本轮验证命令：
  - `cargo test -p vectraparse-core structured_result_round_trip -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers image_metadata_parser_records_decode_failure_bucket -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ffi parse_runtime_records_non_png_image_bucket_before_ocr -- --nocapture`
- 已完成入口识别修复：
  - `vectraparse-mime` 现已识别 `JPEG/TIFF/WebP/GIF/BMP` 的 magic，并支持常见图片扩展名回退。
  - `vectraparse-parsers` 的 `ImageMetadataParser` 已补齐 `GIF/BMP` 支持和签名识别。
  - `vectraparse-ffi` 的 `detect_file` 已使用文件名作为 `resource_name` hint，避免无 magic 时丢失扩展名信息。
- 本轮验证命令：
  - `cargo test -p vectraparse-mime detect_image_magic_and_resource_name_extensions -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers image_metadata_parser_supports_gif_and_bmp_signatures -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ffi parse_runtime_routes_jpeg_bytes_into_image_parser -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ffi detect_file_uses_resource_name_extension_for_images -- --nocapture`
- 已完成 OCR 诊断增强：
  - `vectraparse-ocr` 现已输出 `det_box_count`、`line_count`、`fallback`、`empty_result` 等结构化诊断。
  - `ImageMetadataParser` 现会写入 `image.ocr.enabled`、`image.ocr.error_stage`、`image.ocr.decode_failed`、`image.ocr.box_count`、`image.ocr.line_count`、`image.ocr.fallback`、`image.ocr.empty_result`、`image.ocr.confidence`。
  - parser warning 已按阶段细分为 `image-ocr-decode-failed`、`image-ocr-empty-result`、`image-ocr-low-confidence`，保留 `image-ocr-failed` 作为非 decode 的通用失败桶。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers ocr_success_metadata_records_fallback_and_low_confidence -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers ocr_success_metadata_records_empty_result_warning -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers image_metadata_parser_records_decode_failure_bucket -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ffi parse_runtime_routes_jpeg_bytes_into_image_parser -- --nocapture`
- 已完成轻量图像增强 fallback：
  - `vectraparse-ocr` 在整图和整图英文 fallback 失败后，会继续尝试增强图再做识别。
  - 当前实现实际包含 luma、HSL lightness、max-channel 三类灰度源，每类都生成 `contrast`、`binary`、`binary-invert`、`local-binary`、`local-binary-invert`；`binary` 使用 Otsu 阈值，`local-binary` 使用积分图局部均值阈值。
  - 锐化/去噪仍未单独加入，保留到后续如样本证明有必要时再扩展。
  - 成功命中时会在 diagnostics.fallback 中记录为 `enhanced:<mode>` 或 `enhanced:<mode>:alt`。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr enhancement_variants_include_expected_modes -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr contrast_stretch_expands_low_contrast_range -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr adaptive_binary_can_flip_for_light_foreground -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
- 已完成多尺度与小字策略：
  - `vectraparse-ocr` 在整图增强链路仍为空时，会对“小图”继续尝试 `1.5x` 和 `2x` 上采样整图识别。
  - 当前小图判定使用保守阈值：`width < 640`、`height < 160` 或总像素 `< 160000`。
  - 上采样受到 `MAX_UPSCALE_PIXELS=2500000` 限制，避免大图失败路径上无界放大；成功命中会在 diagnostics.fallback 中记录为 `upscaled:<scale>` 或 `upscaled:<scale>:alt`。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr upscale_variants_generated_for_small_images_only -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr upscale_variants_respect_max_pixel_budget -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr enhancement_variants_include_expected_modes -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
- 已完成文本框后处理改进：
  - `vectraparse-ocr` 现会先对 det 二值 mask 做 1 像素膨胀，再按连通域提取候选框，用原始 det map 的 component score 做过滤。
  - 对小块保守处理：评分不足时回退到原始连通域逻辑，避免细碎高分 blob 被膨胀后误过滤为空。
  - 小框不再额外扩边，较大框才增加轻量 margin；缩放回原图后的扩边逻辑保持不变。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr extract_boxes_from_map_merges_nearby_fragments_after_dilation -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr extract_boxes_from_map_keeps_raw_fallback_for_tiny_high_score_blob -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr extract_boxes_from_map_adds_small_crop_margin -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr enhancement_variants_include_expected_modes -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
- 已部分完成长行与英文行处理：
  - `vectraparse-ocr` 当前有 `dynamic_rec_target_width` 入口，`MAX_REC_IMG_W` 已提升到 `960`，默认配置下长行 crop 可使用更宽的 rec 输入。
  - `recognize_candidate` 保留固定宽度回退：如果动态宽度推理失败，会再用 `cfg.rec_img_w` 执行一次，降低模型动态 shape 不兼容导致整条失败的风险。
  - 检测框裁剪、整图 fallback、增强图 fallback、上采样 fallback、旋转 fallback 和行切分 fallback 现在都会比较中英文两个识别结果。
  - 结果选择优先级基于空结果、ASCII 字符比例和置信度差值，英文行不再只在“中文完全为空”时才有机会走英文模型。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr dynamic_rec_target_width_grows_for_long_lines -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_can_choose_alt_for_ascii_line -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_prefers_primary_when_alt_is_not_clear_win -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr extract_boxes_from_map_merges_nearby_fragments_after_dilation -- --nocapture`
- 已完成本轮识别质量优化：
  - 将 `MAX_REC_IMG_W` 从 `320` 提升到 `960`，长行 crop 可获得更宽 rec 输入；动态宽度推理失败时自动回退到固定宽度。
  - 将全局均值二值化替换为 Otsu 阈值，并新增积分图局部均值二值化增强变体。
  - 增加行级可用性过滤：拒绝低置信、重复字符占比过高、标点占比过高或可读字符比例过低的候选。
- 本轮验证命令：
  - `cargo test -p vectraparse-ocr`（未设置 `ORT_INSTALL_DIR` 时失败：构建脚本找不到 ONNX Runtime）
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
- 已完成旋转保守兜底：
  - `vectraparse-ocr` 在原图结果为空或质量偏低时，会继续尝试 `90/180/270` 三个方向的旋转图。
  - 旋转路径现在会先对旋转图重新跑 det+逐框 rec，命中时 diagnostics.fallback 记为 `det-rotated:<angle>`；仍保留整图识别兜底，命中时记为 `rotated:<angle>` 或 `rotated:<angle>:alt`。
  - 当前未实现小角度 deskew，仍保持在计划边界外，后续是否需要取决于真实失败样本收益。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr rotation_variants_include_expected_angles -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_can_choose_alt_for_ascii_line -- --nocapture`
- 已完成本轮 OCR 质量补强（优化 2/3/5/6）：
  - 增强、上采样和旋转 fallback 现在会先对变体图重新 det+逐框 rec，再保留整图 rec 兜底；变体逐框识别不再对每个 crop 二次展开增强，控制补跑成本。
  - 原图识别结果为空、低置信、det 框多但识别行少、有效字符过少或可读比例偏低时，会触发补跑；补跑结果按行去重合并，明显更长或更高置信时替换原结果。
  - det 后处理增加轻量 unclip 扩边和 NMS，减少裁剪过紧、重复框和重叠框；仍保持轴对齐框。
  - 带透明通道的图片会额外生成黑底 luma/HSL/max-channel 增强变体，用于白字透明底或深色底贴图场景。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers image_metadata_parser_records_decode_failure_bucket`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
  - `git diff --check -- crates/vectraparse-ocr/src/lib.rs docs/dev/4-plan-ocr-preprocessing-optimization.md`
- 已完成区域级布局聚类：
  - `vectraparse-ocr` 将 det+rec 得到的行从全局 y/x 排序升级为区域聚类：先按水平重叠、纵向距离、宽度差异和同行邻近关系归入 `LayoutRegion`，再按区域阅读顺序输出。
  - 多区域输出会在区域之间插入空行，降低复杂截图中左右栏、页头、主内容混排导致的串读概率。
  - OCR diagnostics 新增 `region_count` 和 `layout_applied`，`ImageMetadataParser` 同步写入 `image.ocr.region_count` 与 `image.ocr.layout_applied`。
  - 当前区域聚类只基于几何 bbox，不做语义级版面理解，也不输出独立区域结构体。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers image_metadata_parser_records_decode_failure_bucket`
- 已完成颜色背景区域 fallback：
  - `vectraparse-ocr` 在低质量 fallback 路径中会对图片做粗颜色量化，提取和周边背景不同的连通色块候选，并跳过整页背景、大面积低填充率区域和过小区域。
  - 对颜色区域候选使用边框主色估计背景色，按色差 Otsu 阈值生成黑字白底二值图，再调用 rec；可覆盖部分按钮、标签、卡片标题等“单背景 + 前景文字”的截图区域。
  - 颜色区域候选会转成 `TextLine` 参与现有区域聚类/去重合并；命中时 fallback 记录为 `color-regions` 或 `merged:color-regions`。
  - OCR diagnostics 新增 `color_region_count`，`ImageMetadataParser` 同步写入 `image.ocr.color_region_count`。
  - 当前没有实现无限递归切分；全图粗颜色连通域已经能捕获嵌套色块，后续若真实样本需要，再增加受限深度递归。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
- 已完成结构化区域/行输出和 trace metadata：
  - `OcrResult` 新增 `regions: Vec<OcrTextRegion>` 和 `trace: OcrTrace`；每个 region/line 包含 bbox、text、confidence 和 source。
  - 原图 det、增强图 det、上采样 det、旋转 det、颜色区域和行切分 fallback 会标记来源；上采样和旋转 det 的 bbox 会回映到原图坐标。
  - trace 记录最终选中来源、det pass 次数和 fallback 尝试次数；`ImageMetadataParser` 同步写入 `image.ocr.selected_source`、`image.ocr.det_pass_count`、`image.ocr.fallback_attempt_count`。
  - `VECTRAPARSE_OCR_TRACE=1` 输出同步增加 selected source、det pass 和 fallback attempt，方便复杂截图排查。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
- 已完成 OLE 嵌入图片 OCR 候选处理增强：
  - `vectraparse-mso-binary` 现会在 `.doc` 路径先汇总唯一图片候选，再做 OCR；候选过滤统一使用 OCR 真正支持的图片头判断，并补上 `WebP` 直接流支持。
  - 去重 key 从“长度 + 前 4 字节”提升为“长度 + 前后 64 字节哈希”，降低同源嵌入图被重复 OCR 的概率。
  - OLE 图片 OCR 现在会向上层回传 `ole-image-ocr-budget-hit`、`ole-image-ocr-model-unavailable`、`ole-image-ocr-failed` 等 warning；图片 OCR 失败不会覆盖或中断主文本提取。
  - 复合 payload 切片当前支持 PNG/JPEG/GIF/BMP/DIB；未实现 WebP payload 切片。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary summarizes_doc_image_ocr_candidates_dedups_and_accepts_webp -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary summarizes_doc_image_ocr_candidates_marks_budget_hit -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary extract_text_by_kind_propagates_doc_image_ocr_budget_warning -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary carves_embedded_images_from_stream_payload -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary asserts_empty_content_for_non_text_ole_payload -- --nocapture`
- 已部分完成收口 golden/文档：
  - 将 OCR 相关的稳定样本并入现有提取矩阵：`vectraparse-parsers::extraction_golden_matrix_matches` 现覆盖 `image/png`，用于验证图片 parser 叶子行为和 `image.format` 基础元数据。
  - 同步修复 `ImageMetadataParser` 对 `png` 的格式识别遗漏，并补充显式回归测试。
  - 当前没有纳入真实 OCR 文本 fixture；`image/png` 覆盖的是格式元数据，不证明 OCR 文本识别质量。
  - 收口验证采用分组定向全量测试替代 `cargo test --workspace`：基础 crate 跑 `vectraparse-core`、`vectraparse-mime`；OCR 相关 crate 跑 `vectraparse-ocr`、`vectraparse-parsers`、`vectraparse-ffi`、`vectraparse-mso-binary`。
- 本轮验证命令：
  - `cargo test -p vectraparse-core -p vectraparse-mime`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr -p vectraparse-parsers -p vectraparse-ffi -p vectraparse-mso-binary`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers image_metadata_parser_supports_gif_and_bmp_signatures -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-parsers extraction_golden_matrix_matches -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary extracts_doc_mixed_zh_en_text_sample -- --nocapture`

## 7. Plan 阶段审视

- 安全审查员：
  - 已执行安全门禁；本阶段只新增计划文档和索引，不执行破坏性操作。
  - 后续计划默认避开 `ort.rs` unsafe FFI；若必须修改，需重新分级并补充内存/ABI 验证。
  - 当前证据等级为 E1，计划先补复现和诊断，不直接猜测性改核心逻辑。
- 高级产品：
  - 计划围绕“部分图片无法提取文字”拆解为入口命中、预处理质量、诊断可观测三类问题，避免只调参数。
  - 非目标明确排除模型替换、外部服务和 PDF 渲染 OCR，控制范围。
- 高级架构师：
  - 模块边界保持现有结构：MIME 只负责识别，parser 负责入口和结果，OCR crate 负责图像处理和推理，OLE 只负责嵌入图片候选。
  - 不默认新增依赖；如需新增图像处理依赖须单独确认。
- 高级工程师：
  - 执行顺序先观测、再入口、再预处理，便于每步验证。
  - 每项优化都有定向样本和回退路径，避免一次性重写 OCR 管线。

## 7.1 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs` 和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - `recognize_candidate` 对动态 rec 宽度保留固定宽度回退，降低模型不支持动态 shape 时的失败风险；错误路径仍通过现有 `Result` 返回。
  - 验证证据：`ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr` 通过；parser 侧 decode 失败分桶测试通过。
- 高级产品：
  - 本轮面向可测试的识别质量改进：长行不再固定压缩到 320，低对比/局部阴影增加 Otsu 与局部二值化，明显垃圾候选会被过滤。
  - 未覆盖真实样本质量评估；阈值是否过严需用户用实际图片测试后再调整。
- 高级架构师：
  - 未新增依赖，仍使用现有 `image` crate 和 OCR crate 内部辅助函数。
  - 性能风险主要来自增强变体增加到每个灰度源 5 个；只在 fallback 或 crop 增强路径触发，仍受现有上采样像素和候选数量限制。
- 高级工程师：
  - 变更保持在现有 pipeline 内：rec 宽度、二值化变体、候选质量过滤均为局部函数改动。
  - 单测覆盖新增 Otsu、局部二值化、960 动态宽度和候选过滤；仍缺真实 OCR 文本 fixture。

## 7.2 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs` 和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - 变体 det/rec 失败只影响 fallback 候选，不覆盖原始识别结果；原图为空且基础 rec 失败时仍返回错误，保持原有失败分桶。
  - 验证证据：`cargo test -p vectraparse-ocr`、parser decode 分桶回归、`rustfmt --config skip_children=true --check` 和 `git diff --check` 通过。
- 高级产品：
  - 本轮针对“部分识别但漏主体内容”增加补跑触发条件，并将增强/上采样/旋转变体从单纯整图 rec 升级为先 det+逐框 rec。
  - 透明图黑底增强覆盖白字透明底类样本；实际收益仍需用户真实图片验证。
- 高级架构师：
  - 未新增依赖，继续复用现有 `image` crate；det 后处理仍保持轻量轴对齐框，不引入 polygon/旋转框裁剪复杂度。
  - 性能风险来自低质量路径下变体 det 增多；当前通过仅在 quality fallback 中触发、变体 crop 不再二次增强、上采样像素预算来限制成本。
- 高级工程师：
  - 单测覆盖 alpha 黑底增强、NMS、unclip 扩边、部分成功 fallback 判定和行级去重合并。
  - 仍缺真实 OCR 文本 fixture，无法自动判断这些启发式对具体业务样本的净收益。

## 7.3 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR 文本排序/诊断和 parser metadata 映射，未触碰 `crates/vectraparse-ocr/src/ort.rs`、模型、字典、构建链接或 C ABI。
  - `region_count`、`layout_applied` 是新增 metadata，不改变 OCR 模型输入输出格式；空结果路径保持 `region_count=0`。
- 高级产品：
  - 本轮将复杂截图的输出从“全局行排序”提升为“区域分组后的纯文本输出”，对左右栏、页头+内容区域这类页面截图更友好。
  - 当前仍不输出结构化区域列表，调用方只能通过空行和 metadata 判断是否应用布局聚类。
- 高级架构师：
  - 未新增依赖，区域聚类基于已有 det bbox；没有引入 OCR 模型、版面模型或外部服务。
  - 版面理解停留在几何启发式，后续若要支持表格/卡片/弹窗层级，应新增结构化 block API，而不是继续堆文本拼接规则。
- 高级工程师：
  - 单测覆盖左右两列不串读、全宽页头不吞并下方列区域，以及 parser metadata 记录区域诊断。
  - 仍缺真实复杂截图 fixture，区域阈值需要用用户样本继续校准。

## 7.4 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR fallback 图像预处理、诊断字段和 parser metadata 映射，未触碰 `crates/vectraparse-ocr/src/ort.rs`、模型、字典、构建链接或 C ABI。
  - 颜色区域 fallback 只在低质量路径触发，候选数和处理像素有上限，避免正常 OCR 路径无条件增加成本。
- 高级产品：
  - 本轮针对 UI 截图中“有明显背景色块的文字”增加专项兜底，贴合按钮、卡片、标签、深色/彩色块上的文字场景。
  - 仍需真实截图验证颜色区域是否提升召回，尤其是渐变、阴影和图文混排页面。
- 高级架构师：
  - 未新增依赖，颜色分层使用粗 RGB 量化和连通域；没有引入完整版面分析模型。
  - 当前实现是有限候选生成，不做无限递归；后续若需要递归，应以最大深度、最小区域和候选预算作为边界。
- 高级工程师：
  - 单测覆盖对比色面板区域检测和区域内前景二值化。
  - 仍缺真实 OCR 文本 fixture，`color_region_count` 可用于观察该路径是否命中以及候选规模是否失控。

## 7.5 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只增加 OCR 结果结构和诊断 metadata，未触碰 `crates/vectraparse-ocr/src/ort.rs`、模型、字典、构建链接或 C ABI。
  - 新增 `regions` 是 Rust API 扩展；下游若直接构造 `OcrResult` 需要补字段，本仓库测试已同步。
- 高级产品：
  - 结构化 regions/lines 让复杂截图测试可定位到具体区域和来源，不再只能靠最终纯文本判断。
  - trace metadata 能快速判断是否走了 fallback、是否命中颜色区域、最终采用哪个来源。
- 高级架构师：
  - bbox 坐标保持原图坐标语义；增强图保持 identity，上采样和旋转 det 做坐标回映，整图 fallback 使用原图整框。
  - 当前仍不提供序列化 JSON 字段，parser 侧只沉淀轻量 trace metadata；如上层需要完整区域结构，需要在对应 API 层明确输出格式。
- 高级工程师：
  - 单测覆盖结构化 region/line 的 bbox 和 source；parser metadata 覆盖 selected source、det pass 和 fallback attempt。
  - 仍需真实复杂截图验证旋转 det bbox 回映和区域 source 对齐是否满足调试需求。

## 8. 变更记录

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-06-01 | 创建 OCR 预处理与入口识别优化计划 | 用户要求将优化建议写成 plan |
| 2026-06-01 | 完成执行计划步骤 1：补充最小复现分桶测试与结果解析辅助 | 为后续 OCR 入口识别和预处理优化提供稳定证据 |
| 2026-06-01 | 完成执行计划步骤 2：补齐图片入口识别和 `GIF/BMP` 支持对齐 | 先消除“图片没进入 OCR”主路径问题 |
| 2026-06-01 | 完成执行计划步骤 3：增强 OCR 诊断 metadata/warnings | 让失败样本能稳定分桶到 decode / fallback / empty / low-confidence 等阶段 |
| 2026-06-01 | 完成执行计划步骤 4：增加轻量图像增强 fallback | 为低对比和白字黑底场景增加整图增强识别兜底 |
| 2026-06-01 | 完成执行计划步骤 5：增加多尺度与小字策略 | 为小图和小字号场景增加受限上采样整图识别兜底 |
| 2026-06-01 | 完成执行计划步骤 6：改进文本框后处理 | 通过膨胀合并碎框、component score 过滤和旧逻辑回退提升 det 框稳定性 |
| 2026-06-01 | 部分完成执行计划步骤 7：处理长行与英文行 | 已完成逐 crop 中英文结果选择和 `960` 动态宽度；长 crop 分段仍未实现 |
| 2026-06-01 | 完成执行计划步骤 8：增加旋转保守兜底 | 已完成 90/180/270 度旋转图重新 det+逐框 rec 和整图识别兜底；小角度 deskew 仍未实现 |
| 2026-06-01 | 完成执行计划步骤 9：增强 OLE 嵌入图片 OCR 候选处理 | 通过 `.doc` 候选去重、预算告警和失败诊断降低嵌入图 OCR 对主提取链路的干扰 |
| 2026-06-01 | 部分完成执行计划步骤 10：收口 golden/文档 | 已覆盖 `image/png` parser 元数据；真实 OCR 文本 golden 尚未落地 |
| 2026-06-01 | 修复 PNG OCR 回归：统一 rec 白底预处理并移除 det 无框伪整图框 | 针对普通 RGBA PNG 无法识别的回归，恢复 det/rec 图像前处理一致性并让整图 fallback 只在真实 fallback 阶段触发 |
| 2026-06-01 | 按当前代码实现校准 OCR 计划文档 | 移除与代码不一致的 960 rec 宽度、真实 OCR golden、旋转逐框识别等完成表述 |
| 2026-06-01 | 完成本轮 OCR 识别质量优化 | 放开长行 rec 宽度到 960，增加 Otsu/局部二值化增强变体，并增加行级候选质量过滤 |
| 2026-06-01 | 完成 OCR 质量补强优化 2/3/5/6 | 对增强/上采样/旋转变体重跑 det，增加部分成功补跑合并、det NMS/unclip 扩边和透明图黑底增强 |
| 2026-06-01 | 完成区域级布局聚类 | 将复杂截图 OCR 输出从全局 y/x 排序升级为区域聚类输出，并增加区域诊断 metadata |
| 2026-06-01 | 完成颜色背景区域 fallback | 针对明显背景色块上的文字增加色块候选、区域二值化识别和颜色区域诊断 metadata |
| 2026-06-01 | 完成结构化 OCR regions/lines 和 trace metadata | 让复杂截图 OCR 结果可按区域、行、来源和 fallback 路径调试 |
