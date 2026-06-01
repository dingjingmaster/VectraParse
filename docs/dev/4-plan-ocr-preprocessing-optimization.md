# OCR 预处理与入口识别优化开发计划

> 文档元数据
> - 文件编号：4
> - 文档类型：plan
> - 文件路径：docs/dev/4-plan-ocr-preprocessing-optimization.md
> - 文档版本：v1.0.0
> - 最后更新：2026-06-01
> - 关联需求：分析并规划 OCR 预处理优化，解决部分图片无法提取文字的问题。
> - 关联调研：当前为代码只读分析结论，未单独创建 research 文档。

## 1. 目标与成功标准

- 任务目标：
  - 提升图片 OCR 入口命中率，避免 JPEG/TIFF/WebP/BMP/GIF 等图片因 MIME 识别不足而绕过 OCR。
  - 提升 OCR 对低分辨率、小字、低对比、白字黑底、长行、英文行、倾斜/旋转图片的文本提取成功率。
  - 增强 OCR 失败路径可观测性，能区分未进入 OCR、解码失败、检测无框、识别为空和低置信度输出。
- 成功标准：
  - 常见图片格式能稳定进入 `ImageMetadataParser` 的 OCR 路径。
  - 失败样本可按原因分桶，并在 metadata 或 warnings 中体现。
  - 新增 OCR golden/fixtures 覆盖至少：PNG、JPEG、小字截图、低对比图、白字黑底、长英文行、旋转图。
  - 定向测试通过，并能用失败样本证明优化前后的差异。
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
  - OCR 核心：增加图像增强、多尺度、长行处理、语言模型选择和诊断。
  - OLE 嵌入图：增强候选图片过滤、去重和 OCR 失败诊断。
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
- 根因假设：
  - 非 PNG 图片可能未被 MIME 识别为图片，直接绕过 OCR。
  - 预处理仅灰度缩放和平均亮度反色，无法覆盖低对比、小字、暗色 UI、长行、倾斜/旋转等场景。
  - det 后处理使用简化连通域矩形，缺少 dilation/unclip/box score/旋转框，易漏框或裁剪不完整。
  - 英文模型只在整图识别为空时兜底，混合中英文或英文行可能被中文模型误识别后不再重试。
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
| 4 | 增加轻量图像增强 fallback：灰度对比度拉伸、局部/全局阈值、白字黑底双路径、锐化或去噪的保守组合 | `cargo test -p vectraparse-ocr`; 低对比/暗色 UI fixture 输出非空且已知样本不回退 | 完成 |
| 5 | 增加多尺度与小字策略：对低分辨率或 det 空结果执行 1.5x/2x 上采样，限制最大像素和重试次数 | 定向 OCR 样本测试；记录耗时和内存上限，避免大图性能失控 | 完成 |
| 6 | 改进文本框后处理：det map 做 dilation/box score 过滤/扩边，减少碎框和裁剪不完整；保留旧逻辑 fallback | OCR det 单元测试或 fixture 对比；验证多行截图顺序稳定 | 完成 |
| 7 | 处理长行与英文行：长 crop 分段或动态 rec 宽度；每个 crop 可按置信度/字符集选择中文或英文模型 | 长英文行、混合中英文截图 fixture；确认没有大量乱码输出 | 完成 |
| 8 | 增加旋转/倾斜保守兜底：先支持 90/180/270 度方向重试；小角度 deskew 视样本收益决定是否落地 | 旋转 fixture；验证正常方向图片不会额外引入重复文本 | 完成 |
| 9 | 增强 OLE 嵌入图片 OCR 候选处理：格式过滤、去重 key、预算命中诊断，确保嵌入图失败不影响主文本 | `cargo test -p vectraparse-mso-binary`; OLE 样本保持可提取 | 完成 |
| 10 | 收口 golden/文档：补 OCR golden 样本、更新验证说明，记录未覆盖场景和后续边界 | `cargo test --workspace` 或定向替代命令；更新 `docs/dev/4-plan...` 后续状态/总结 | 完成 |

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
  - 没有真实失败样本前，只能用构造样本覆盖典型场景，不能证明所有生产失败图已修复。
- 残余风险：
  - 预处理增强可能提升召回但带来误识别和重复文本，需要用置信度、字符集检查和 fallback 选择控制。
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
  - `vectraparse-ocr` 在整图和整图英文 fallback 失败后，会继续尝试 `contrast`、`binary`、`binary-invert` 三种保守增强图再做识别。
  - 当前实现覆盖低对比和白字黑底双路径；锐化/去噪仍未单独加入，保留到后续如样本证明有必要时再扩展。
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
- 已完成长行与英文行处理：
  - `vectraparse-ocr` 现按 crop 长宽比动态放大 `rec` 输入宽度，最长可到 `960`，避免长行被固定压缩到 `320`。
  - 检测框裁剪、整图 fallback、增强图 fallback、上采样 fallback 和行切分 fallback 现在都会比较中英文两个识别结果。
  - 结果选择优先级基于空结果、ASCII 字符比例和置信度差值，英文行不再只在“中文完全为空”时才有机会走英文模型。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr dynamic_rec_target_width_grows_for_long_lines -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_can_choose_alt_for_ascii_line -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_prefers_primary_when_alt_is_not_clear_win -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr extract_boxes_from_map_merges_nearby_fragments_after_dilation -- --nocapture`
- 已完成旋转保守兜底：
  - `vectraparse-ocr` 在整图、增强图和上采样 fallback 仍为空后，会继续尝试 `90/180/270` 三个方向的整图识别。
  - 旋转路径同样会走逐 crop 的中英文结果选择逻辑；命中时 diagnostics.fallback 记为 `rotated:<angle>` 或 `rotated:<angle>:alt`。
  - 当前未实现小角度 deskew，仍保持在计划边界外，后续是否需要取决于真实失败样本收益。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr rotation_variants_include_expected_angles -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr default_result_is_empty -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-ocr select_recognition_can_choose_alt_for_ascii_line -- --nocapture`
- 已完成 OLE 嵌入图片 OCR 候选处理增强：
  - `vectraparse-mso-binary` 现会先汇总唯一图片候选，再做 OCR；候选过滤统一使用 OCR 真正支持的图片头判断，并补上 `WebP` 直接流支持。
  - 去重 key 从“长度 + 前 4 字节”提升为“长度 + 前后 64 字节哈希”，降低同源嵌入图被重复 OCR 的概率。
  - OLE 图片 OCR 现在会向上层回传 `ole-image-ocr-budget-hit`、`ole-image-ocr-model-unavailable`、`ole-image-ocr-failed` 等 warning；图片 OCR 失败不会覆盖或中断主文本提取。
- 本轮验证命令：
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary summarizes_doc_image_ocr_candidates_dedups_and_accepts_webp -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary summarizes_doc_image_ocr_candidates_marks_budget_hit -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary extract_text_by_kind_propagates_doc_image_ocr_budget_warning -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary carves_embedded_images_from_stream_payload -- --nocapture`
  - `LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 cargo test -p vectraparse-mso-binary asserts_empty_content_for_non_text_ole_payload -- --nocapture`
- 已完成收口 golden/文档：
  - 将 OCR 相关的稳定样本并入现有提取矩阵：`vectraparse-parsers::extraction_golden_matrix_matches` 现覆盖 `image/png`，用于验证图片 parser 叶子行为和 `image.format` 基础元数据。
  - 同步修复 `ImageMetadataParser` 对 `png` 的格式识别遗漏，并补充显式回归测试。
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
| 2026-06-01 | 完成执行计划步骤 7：处理长行与英文行 | 通过动态 rec 宽度和逐 crop 中英文结果选择提升长英文行识别稳定性 |
| 2026-06-01 | 完成执行计划步骤 8：增加旋转保守兜底 | 通过 90/180/270 度整图重试覆盖横竖方向错误场景 |
| 2026-06-01 | 完成执行计划步骤 9：增强 OLE 嵌入图片 OCR 候选处理 | 通过候选去重、预算告警和失败诊断降低嵌入图 OCR 对主提取链路的干扰 |
| 2026-06-01 | 完成执行计划步骤 10：收口 golden/文档 | 通过稳定图片样本并入提取矩阵、分组全量测试和边界记录完成本轮 OCR 优化闭环 |
