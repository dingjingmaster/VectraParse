# OCR 预处理与入口识别优化开发计划

> 文档元数据
> - 文件编号：4
> - 文档类型：plan
> - 文件路径：docs/dev/4-plan-ocr-preprocessing-optimization.md
> - 文档版本：v1.0.23
> - 最后更新：2026-06-02
> - 关联需求：按当前代码实现校准 OCR 预处理优化计划与完成状态；按识别质量优化建议实现一版可测试改进；继续实现增强/上采样/旋转重跑 det、部分成功补跑合并、det NMS/unclip 扩边、透明图黑底增强、复杂截图区域级布局聚类、颜色背景区域 fallback、结构化 regions/lines 输出、trace metadata、det 框内部多行切分、浅色底黑字主动补充识别、det 候选质量优化、rec tight crop、近似重复去重、短噪声过滤、page-region 局部修复预算、高分辨率 tile det 补充、未覆盖纹理区域补识别、二维视觉区域候选、行级候选合并、宽行分段 rec、颜色背景区域局部 det 补识别、低对比局部二值化兜底、大框强制结构化拆分、tile 本地拆分预算、低对比前景 mask 拆行拆列、补充候选优先级排序、det 合并视觉分隔保护、近重复候选投票仲裁、全局候选池仲裁、分层背景区域识别、局部多预处理/多尺度 det 补充、OCR 文本 golden 规则、CTC margin、候选池评分细化、视觉边界版面分组、轻量 deskew、自适应局部预算、模型候选选择优化、行级 margin 仲裁、超大框先拆后识别、宽行滑窗识别、panel 分桶版面、颜色层级前景组件、低质 ASCII 噪声过滤、CTC 小 beam 解码、CTC prefix beam、det 轮廓投影细化、trace margin、文本行图版面、glyph textness、候选质量校准、OCR trace 指标规则、主 det 图像上下文版面、候选 trace 事件、空结果视觉补扫、dominant 背景软前景、保守 det 合并、长行动态宽度/分段、CTC 概率校准、候选池行级 support 仲裁、递归 panel-first 区域识别、低阈值局部 det、多前景 mask 融合、字符级 trace 指标和 trace 指标/metadata 回归入口。
> - 关联调研：当前为代码只读分析结论，未单独创建 research 文档。

## 1. 目标与成功标准

- 任务目标：
  - 提升图片 OCR 入口命中率，避免 JPEG/TIFF/WebP/BMP/GIF 等图片因 MIME 识别不足而绕过 OCR。
  - 提升 OCR 对低分辨率、小字、低对比、白字黑底、长行、英文行、倾斜/旋转图片的文本提取成功率。
  - 增强 OCR 失败路径可观测性，能区分未进入 OCR、解码失败、检测无框、识别为空和低置信度输出。
- 成功标准：
  - 常见图片格式能稳定进入 `ImageMetadataParser` 的 OCR 路径。
  - 失败样本可按原因分桶，并在 metadata 或 warnings 中体现。
  - 当前单元测试覆盖 MIME 入口、诊断 metadata/warnings、图像增强变体、小图上采样、det mask 后处理、中英文结果选择、旋转变体、行级质量过滤、区域级布局聚类、颜色背景区域候选、浅色底黑字候选、det 原始小框替代切分候选、rec 前景 tight crop、近似重复行去重、短 ASCII/符号噪声过滤、结构化区域/行输出、det 框内部多行切分、行级 margin 投票、超大框优先结构化拆分、宽行滑窗、panel 分桶、颜色前景组件、CTC prefix beam、det 轮廓投影细化、trace margin、文本行图版面、glyph textness、候选质量校准、候选 trace 事件、空结果视觉补扫、dominant 背景软前景、保守 det 合并、长行动态分段、CTC 概率校准、候选 support_count、递归 panel 子区域、低阈值局部 det、多前景 mask 变体和字符级 trace 指标，并覆盖 OLE 图片候选处理。
  - 当前 golden 覆盖 `image/png` parser 叶子行为、`image.format` 基础元数据和 OCR trace 文本/指标规则入口；尚未纳入私有真实截图 OCR 文本 fixture。
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
  - OCR 核心：当前实现包含原图 det+逐框 rec、主 det 图像上下文版面分组、原始 det 小框作为合并框的替代切分候选、det map 膨胀连通域和轮廓投影细化、det 合并视觉分隔保护和更保守的小 panel 间隙保护、rec 前景 tight crop、CTC top1/top2 margin 统计、CTC prefix beam 备选解码、CTC 帧级概率校准、字符级最小置信度统计、det 框内部多行 crop 行投影切分、同一检测行内水平大空隙切分、大框强制结构化拆分、超大框先拆后 direct rec、宽行窄空隙分段 rec、超长行动态分段预算、宽行无可靠切点时的重叠滑窗 rec、聊天 UI 中间时间标记断行后处理、文本行图版面聚类、视觉边界辅助区域分组、panel 分桶版面聚类、递归 panel-first 区域识别、bbox+文本相似度的近似重复行去重、近重复候选投票仲裁、行级 margin 质量评分、行级 support_count 仲裁、全局候选池仲裁、候选 trace 事件和 candidate_count、候选采用/拒绝统计、候选来源权重和多来源支持评分、glyph textness 候选评分、短 ASCII/符号噪声过滤、低置信异常 ASCII token 过滤、颜色背景区域候选 fallback、受限主动颜色区域补充识别、分层背景区域识别、dominant 背景软前景 mask、颜色 panel 内前景组件候选、补充候选优先级排序、颜色背景区域局部 det 补识别、颜色区域局部放大 det 补充、颜色区域低对比局部二值化兜底、空结果视觉前景补扫、局部多窗口二值化 rec 候选、多前景 mask 融合、自适应局部预处理预算、低对比前景 mask 拆行/拆列、4-bit 颜色量化的浅色底黑字候选、未覆盖纹理区域补识别、二维视觉区域候选、行级候选合并、page-region 本地拆行/修复预算、高分辨率 tile det 补充和 tile 本地拆行/修复预算、局部低阈值 det 补扫、结构化 `regions/lines` 输出、trace 统计、trace source/candidate action 计数、trace 行级 `support_count/readable_ratio/char_min_confidence`、整图 fallback、轻量小角度 deskew 整图 fallback、luma/HSL/max-channel 的对比度拉伸、Otsu 全局二值化、局部均值二值化、透明图黑底增强变体、小图 1.5x/2x 上采样 fallback、90/180/270 度旋转 fallback、增强/上采样/旋转图重新 det+逐框 rec、行切分 fallback、中英文 rec 结果选择、基于 CTC margin 的模型候选选择、行级质量过滤、部分成功质量判定和诊断。
  - OCR 核心未实现：锐化/去噪、真实私有截图 OCR 文本 fixture；det 后处理仍输出轴对齐框，未实现真正 PaddleOCR polygon unclip、旋转框裁剪或语义级页面版面理解；小角度 deskew 仅作为整图 fallback，不对 det bbox 做任意角度回映；宽行滑窗只在修复路径中保守触发，不做语义级重排或词典纠错。
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
  - 中英文 rec 已在逐 crop 和各类 fallback 中统一比较；当前动态 rec 宽度上限为 `MAX_REC_IMG_W=640`，并保留固定宽度失败回退，避免模型不接受宽输入时整条识别失败；此前 960 上限在复杂整页截图上成本过高，已收紧。
  - fallback 已从“仅空结果触发”扩展为“空结果、低置信、det 框多但识别行少、有效字符过少或可读比例偏低”触发；其中“det 框多但识别行少”会跳过高置信、可读且已有足够文本的结果，避免复杂截图无谓进入重 fallback；补跑候选会按行去重后合并，或在明显更长/更高置信时替换原结果。
  - 检测框识别结果已不再只按全局 y/x 排序；当前会按 bbox 水平重叠、纵向距离、宽度差异将行聚类为区域，区域之间用空行分隔，并在 metadata 中记录 `image.ocr.region_count` 和 `image.ocr.layout_applied`。
  - 原始 det 小框不会直接追加到最终结果；当前只作为合并宽框的替代切分候选，候选切分必须通过内容量和置信度门禁才替代整框识别，避免召回过头造成重复噪声。
  - rec 输入前会尝试根据前景 mask 裁掉大块空白并保留 padding，降低宽 crop 或空白背景对 rec 缩放和识别的干扰。
  - 行级输出会先做 bbox 重叠 + 文本相似度去重，并对同列长文本近似重复、全局完全相同长文本做保守去重；近重复候选会先聚成簇，再按基础质量、精确文本支持、近似文本支持和来源支持投票选出代表行；密集结果中的短 ASCII/符号行会被过滤，短中文名字等可能真实重复的文本不做全局强去重。
  - 颜色背景区域识别会在正常 det 后主动用小预算补充一次，并在低质量路径中继续作为 fallback；当前颜色量化提升到 4-bit，可区分浅灰/浅蓝底和页面背景，候选区域内会根据区域主色做前景色差二值化，再执行 rec；主动补充只合并与已有 det 行不明显重叠的候选，避免重复噪声，并在 metadata 中记录 `image.ocr.color_region_count`。
  - 颜色区域、未覆盖视觉区域和颜色区域局部 det 的补充候选会按文本纹理密度、前景面积、框面积惩罚和大框可拆性排序，预算优先消耗在更像文字且更高收益的候选上。
  - 全局候选池会把 det、page-region、tile、颜色区域、分层区域、视觉区域和 fallback 产生的结构化行统一合并，再走行级去重、近重复投票和区域聚类；避免单一路径候选因单独看增量不足而无法补入最终结果。
  - 分层背景区域识别会先用颜色/视觉 panel 候选定位背景块，再在 panel 内用前景 mask 做行投影；如果 panel 内还有子背景色块，会递归提取子区域文字候选。该路径不做业务词匹配，候选最终仍受文字纹理评分、可靠重叠过滤和识别预算限制。
  - 局部 crop 识别在 direct rec 之外增加前景二值化、局部小/中/大窗口二值化及反色窗口候选；颜色区域局部 det 对选中的 panel crop 增加受限局部 2x/1.5x 放大 det pass，并把 bbox 映射回原图坐标。
  - CTC 解码会记录被输出字符的 top1/top2 margin，主/备模型选择和修复候选比较会把 margin 纳入质量评分；这属于模型输出解释和候选选择优化，不替换 ONNX 模型文件。
  - 全局候选池和近重复行投票会加入来源可靠度和多来源支持评分；原图 det、page/tile、颜色局部 det、分层/视觉区域等候选不会被完全等权处理。
  - 区域聚类在有原图上下文时会检查水平/垂直低纹理空白和分隔线，避免几何上接近但视觉上被 panel/gutter 分开的文本被合并。
  - 局部多窗口二值化变体现在按 direct rec 质量自适应触发：高置信且 margin 稳定的 crop 不继续跑局部变体，中等结果只跑有限变体，弱结果才跑完整局部候选。
  - OCR trace golden 脚本已支持最终文本 `full_text`、`text_contains`、`text_not_contains` 规则，可作为真实截图文本 golden 的入口；当前仓库样例仍使用合成 `Alpha/Beta` trace，未纳入私有真实截图内容。
  - 颜色背景区域现在会对存在前景信号、且不是单行已可靠覆盖的背景块做受限局部 det；局部 det 结果只作为强 supplement 行合并，避免整块 rec 失败时漏掉面板内多行或小字。
  - 颜色区域前景二值化在全局色差不足或色差前景比例异常时，会尝试局部低对比二值化兜底；兜底结果必须满足前景比例和最小前景范围门禁，纯色低对比块不会被当成文字。拆行、拆列、tight crop、tile 评分和颜色区域 det 候选评分现在也复用低对比前景 mask，降低浅底深色弱对比文字在结构化拆分前漏掉的概率。
  - `OcrResult` 已输出结构化 `regions: Vec<OcrTextRegion>`，每个 region 包含 bbox、文本、置信度、来源和行级 `OcrTextLine`；trace 当前记录最终选中来源、det pass 次数和 fallback 尝试次数，并通过 metadata 写出。
  - 原图 det 框识别前会优先尝试明显背景色块切分，再对包含多行文字的 crop 做前景行投影切分，并对同一行内存在水平大空隙的宽 crop 做列段切分；只有拆出至少两条有效子段且子段识别未明显丢内容/置信度时，才用子段结果替代整框 rec，避免相邻消息或左右区域被 rec 模型粘成一行；每次 det pass 额外子行 rec 和逐 crop 增强都有固定预算，子段只做 direct rec，增强/旋转 det 不再叠加拆行成本。det 框合并时会检查相邻框之间是否存在足够宽的低纹理沟槽或竖向视觉分隔，降低跨面板/卡片误合并概率。
  - page-region 重新 det 路径现在也拥有独立的小型拆行/修复预算，用于修复区域 crop 内的低质大框或空识别行；高分辨率 tile det 补充只在大图且当前结果为空、低置信、det 框多但有效行少或已有行可修复时触发，并受 `MAX_HIGH_RES_TILE_DET_PASSES` 限制；tile source 现在拥有有限拆行/修复预算，避免 tile 内整块面板直接送入 rec。
  - 原图、page-region、color-region-det 和 tile 中的超大检测框会在 direct rec 后尝试结构化拆分；拆分候选必须通过原有 split 内容/置信门禁，或在 direct 结果本身需要修复且拆分文本质量明显更高时才替代整框识别，避免为了召回牺牲已可靠单行。
  - 未覆盖纹理区域补识别会在已有可靠 OCR 行之外寻找前景/边缘文字纹理区域，用受限预算进行二值化或直接 rec；页面视觉区域候选已从单纯列投影扩展到二维面板候选，候选合并优先按结构化 line 去重聚类。
  - 针对聊天截图里“发送者：预览文本 + 刚刚/昨天/星期X + 另一侧内容”被 rec 串成一行的情况，识别后会在中间时间标记前断行；该规则要求前缀包含发送者分隔符且前后都有足够文本，避免普通句子误切。
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
| 7 | 处理长行与英文行：每个 crop 可按置信度/字符集选择中文或英文模型；动态 rec 宽度上限当前为 `640` 并保留固定宽度失败回退；宽行在存在可靠低前景切点时可分段 rec | 当前测试覆盖中英文结果选择、640 宽度行为和宽行分段辅助函数；长英文真实 OCR fixture 未落地 | 部分完成 |
| 8 | 增加旋转保守兜底：支持 90/180/270 度旋转图重新 det+逐框 rec，并保留整图识别重试；小角度 deskew 后续在步骤 27 作为整图 fallback 补入 | 当前测试覆盖旋转变体生成；未覆盖旋转真实 OCR 文本 fixture | 完成 |
| 9 | 增强 OLE 嵌入图片 OCR 候选处理：仅 `.doc` 路径做图片候选汇总、格式过滤、去重 key 和预算命中诊断，嵌入图失败不影响主文本 | `cargo test -p vectraparse-mso-binary`; 当前 WebP 支持直接流，payload 切片未实现 | 完成 |
| 10 | 收口 golden/文档：当前已将 `image/png` stub 纳入 parser 提取矩阵，并在步骤 27 补入 OCR trace/text golden 规则入口；私有真实截图文本样本尚未进入 golden | 定向替代命令；更新 `docs/dev/4-plan...` 后续状态/总结 | 部分完成 |
| 11 | 增加区域级布局聚类：逐框识别后按 bbox 聚类为区域，区域内按行排序，区域间用空行分隔，并输出区域诊断 metadata | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 12 | 增加颜色背景区域 fallback：按近似背景色连通域提取 UI 色块，在候选区域内做背景色差二值化后识别，并输出颜色区域候选数 metadata | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 13 | 增加结构化区域/行输出和 trace metadata：暴露 OCR region/line bbox、source、confidence，并记录选中来源、det pass 和 fallback 尝试数 | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 14 | 增加 det 框内部多行/多段切分：对原图单个检测框内的背景色块、前景行投影和行内水平大空隙做保守切分，拆分结果通过内容量和置信度门禁后作为多条 TextLine 输出，并限制额外子行 rec 和逐 crop 增强预算 | `cargo test -p vectraparse-ocr`; `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs` | 完成 |
| 15 | 增加聊天 UI 时间标记断行后处理：对发送者前缀后串入 `刚刚/昨天/星期X` 的合并行插入换行 | `cargo test -p vectraparse-ocr`; 用户提供复杂截图样本实测 | 完成 |
| 16 | 增加浅色底黑字主动补充识别：正常 det 后用小预算识别颜色区域候选，颜色量化提升到 4-bit 以捕获浅灰/浅蓝 UI 色块 | `cargo test -p vectraparse-ocr`; parser metadata 定向测试 | 完成 |
| 17 | 增加 det 候选质量优化、rec 前景 tight crop 和近似重复去重：原始小框仅作为合并框替代切分候选，rec 前裁掉大块空白，输出前按 bbox/text 相似度去重 | `cargo test -p vectraparse-ocr`; parser metadata 定向测试；真实截图回测 | 完成 |
| 18 | 增加短噪声过滤和全局长文本精确去重：密集结果中过滤短 ASCII/符号噪声，完全相同或近乎完全相同的长文本可跨位置去重 | `cargo test -p vectraparse-ocr`; parser metadata 定向测试；真实截图回测 | 完成 |
| 19 | 增加 page-region 本地修复和高分辨率 tile det 补充：区域 crop 内保留受限拆行/修复预算，大图低质或空结果时按纹理密度选取有限 tile 用原图分辨率重新 det/rec | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 20 | 增加未覆盖纹理区域补识别、二维视觉区域候选和行级候选合并：已有可靠行之外的纹理区域可受限补 rec，视觉 page-region 可切 2D 面板，fallback 合并优先用结构化 line 去重 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 21 | 增加宽行分段 rec：对需要修复的超宽/高宽比异常文本框，在存在可靠低前景切点时按段识别并拼接，只有优于 direct 结果才采纳 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 22 | 增加颜色背景区域局部 det 补识别：对存在前景信号的颜色背景块做有限 crop det/rec，单行已可靠覆盖的背景块跳过，补充行通过强候选和重叠过滤后合并 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 23 | 增加低对比局部二值化兜底：颜色区域全局色差不足或前景比例异常时，用局部均值阈值提取深色文字，并拒绝纯色低对比块 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 24 | 增加大框强制结构化拆分、tile 本地拆分/修复预算和低对比前景 mask 拆行拆列：超大检测框在 direct rec 后可受限拆分，tile 内大框也可局部修复，弱对比文字可参与行/列投影 | `cargo test -p vectraparse-ocr`; 真实复杂截图实测；trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 25 | 增加补充候选优先级排序、det 合并视觉分隔保护和近重复候选投票仲裁：预算优先给更像文字的候选，检测框合并避开明显视觉沟槽，多路径近重复文本按支持度选择代表行 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 26 | 增加全局候选池、分层背景区域识别和局部多预处理/多尺度 det 补充：多路径候选统一仲裁，panel 内前景行可递归补识别，局部 crop 可用多窗口二值化和局部放大 det 提升召回 | `cargo test -p vectraparse-ocr`; trace golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 27 | 一次性完成后续准确性优化 1-7：扩展 OCR 文本 golden 规则、增加 CTC margin、细化候选池评分、让视觉边界参与版面分组、增加轻量 deskew、实现局部自适应预算，并用 margin 优化主/备模型候选选择 | `cargo test -p vectraparse-ocr`; trace/text golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 28 | 一次性完成新一轮准确性优化 1-7：行级 margin 质量仲裁、超大框先拆后识别、宽行重叠滑窗、panel 分桶版面、颜色层级前景组件、低质 ASCII 噪声过滤和 CTC 小 beam 备选解码 | `cargo test -p vectraparse-ocr`; trace/text golden；`cargo check -p vectraparse-ffi`; 格式和 diff 检查 | 完成 |
| 29 | 一次性完成准确性优化 1-7：CTC prefix beam 概率合并、det 轮廓投影细化、trace margin 暴露、文本行图版面、glyph textness、结构化候选评分校准和 OCR trace 指标规则扩展 | `cargo test -p vectraparse-ocr`; OCR trace golden；parser 定向测试；`cargo check -p vectraparse-ffi`; 格式检查 | 完成 |
| 30 | 一次性完成准确性优化 1-7：主 det 图像上下文版面、候选 trace 事件、空结果视觉补扫、dominant 背景软前景、保守 det 合并、长行动态宽度/分段和 CTC 概率校准 | `cargo test -p vectraparse-ocr`; OCR trace golden；parser 定向测试；`cargo check -p vectraparse-ffi`; 格式检查 | 完成 |
| 31 | 一次性完成准确性优化 1-7：候选池行级 support 仲裁、递归 panel-first 区域识别、低阈值局部 det、多前景 mask 融合、字符级 trace 指标和 trace 指标/metadata 回归入口 | `cargo test -p vectraparse-ocr`; OCR trace golden；parser 定向测试；`cargo check -p vectraparse-ffi`; `git diff --check` | 完成 |

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
  - 当前有 OCR trace/text golden 规则入口，但没有纳入真实私有截图 fixture，无法自动发现长行、小字、旋转、低对比场景的业务样本质量回退。
- 残余风险：
  - 预处理增强和变体重跑 det 可能提升召回但带来误识别、重复文本和耗时增加；当前已增加低置信、重复字符、高标点占比过滤、NMS 和行级去重合并，但仍缺少真实样本阈值校准。
  - 行级质量过滤可能减少垃圾输出，也可能丢弃少量真实低置信文本；需要用户测试样本评估阈值是否过严。
  - 部分成功 fallback 的替换/合并策略仍是启发式：可能保留少量原始漏识别片段，也可能在候选更长时替换掉原有短文本。
  - 区域级布局聚类仍是几何启发式，不理解页面语义；复杂表格、嵌套卡片、瀑布流、多弹窗叠加或跨区域标题仍可能排序不符合人工阅读顺序。
  - 颜色背景区域识别使用 4-bit 颜色量化和区域主色背景估计；渐变、阴影、图片背景、透明叠层或彩色插图可能产生候选过多或候选缺失，当前主动路径限制识别候选预算，低质量 fallback 路径仍限制最大候选数。补充候选优先级排序是启发式，可能把少量真实稀疏文本排到较后位置，需要真实样本继续调阈值。
  - 结构化 bbox 对原图 det、增强图 det、上采样和旋转 det 做了坐标回映；复杂旋转/裁剪组合仍需真实截图验证坐标是否足够准确。
  - det 框内部多行/多段切分依赖背景色块、前景/背景估计、行投影和列投影，能覆盖相邻消息被同一检测框包住、或同一 y 行跨左右区域串读的场景；如果行距/列距极小、背景复杂、前景比例异常，或额外子行 rec 超出预算，仍可能退回整框识别。大框强制拆分和 tile 本地预算会增加少量 rec 尝试，复杂截图仍需用真实样本观察耗时。det 合并视觉分隔保护可能保留更多相邻短框，后续仍需在真实样本上评估是否影响同一自然语言行的完整性。
  - 全局候选池是召回优先的结构化合并策略，虽然仍经过行级质量、去重和区域聚类，但在候选本身误识别且字符数增加时可能引入额外噪声。
  - 分层背景区域识别和局部放大 det 会增加有限数量的 crop rec/det 尝试；复杂页面 panel 很多时仍可能抬高耗时，需要真实样本继续观察预算是否过宽。
  - 局部多窗口二值化可提升弱对比文字召回，也可能让阴影、边框或图标更像文字前景；当前依赖候选质量和 `is_usable_recognition` 过滤，仍需真实截图校准。
  - CTC margin 是模型输出解释启发式，不等同于语义正确性；高 margin 的错字仍可能发生。
  - 轻量 deskew 只对整图 fallback 生成纠偏图，不做任意角度 det bbox 回映；如果纠偏图识别被采纳，结构化 bbox 只能保守使用整图范围。
  - 视觉边界辅助版面分组依赖低纹理/分隔线检测，复杂纹理背景或非常细的分割线仍可能漏判或误判。
  - 宽行滑窗会提高连续长行召回，但窗口重叠合并仍依赖短文本重叠，存在少量重复或断字风险。
  - 低质 ASCII token 过滤只在低置信和异常大小写/短 token 场景触发，仍可能误伤少量真实低置信英文短词，需要真实样本校准。
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
  - `vectraparse-ocr` 当前有 `dynamic_rec_target_width` 入口，`MAX_REC_IMG_W` 当前为 `640`，默认配置下长行 crop 可使用比固定 320 更宽的 rec 输入。
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
  - 将 `MAX_REC_IMG_W` 从最初固定 `320` 放宽到当前 `640`，长行 crop 可获得更宽 rec 输入；动态宽度推理失败时自动回退到固定宽度。
  - 将全局均值二值化替换为 Otsu 阈值，并新增积分图局部均值二值化增强变体。
  - 增加行级可用性过滤：拒绝低置信、重复字符占比过高、标点占比过高或可读字符比例过低的候选。
- 本轮验证命令：
  - `cargo test -p vectraparse-ocr`（未设置 `ORT_INSTALL_DIR` 时失败：构建脚本找不到 ONNX Runtime）
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
- 已完成旋转保守兜底：
  - `vectraparse-ocr` 在原图结果为空或质量偏低时，会继续尝试 `90/180/270` 三个方向的旋转图。
  - 旋转路径现在会先对旋转图重新跑 det+逐框 rec，命中时 diagnostics.fallback 记为 `det-rotated:<angle>`；仍保留整图识别兜底，命中时记为 `rotated:<angle>` 或 `rotated:<angle>:alt`。
  - 历史步骤 8 当时尚未包含小角度 deskew；步骤 27 已补入轻量整图 deskew fallback，后续是否继续扩大到检测框级 deskew 取决于真实失败样本收益。
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
  - `VECTRAPARSE_OCR_TRACE=1` 输出同步增加 start、det pass 进度、selected source、det pass 和 fallback attempt，方便复杂截图排查耗时和路径。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
- 已完成 det 框内部多行/多段切分：
  - `recognize_detected_text` 在原图 det bbox 内先尝试背景色块切分，再尝试前景行投影切分；若单行宽 crop 中存在明显水平大空隙，会继续切成多个列段；切分候选会独立 rec，并以 `det:split` source 标记。
  - 为避免复杂截图和 fallback 变体放大耗时，框内拆行只在原图 `det` pass 启用，并限制每次 det pass 最多额外识别 6 条子行；拆出的子段先做 direct rec，失败时只对该子段做背景色二值化 rec，若子段可用则直接采用，不再先跑整框 rec 做比较；每次 det pass 最多允许 4 个中小弱 crop 进入增强识别，大 crop 和其余弱框保留 direct 结果；质量 fallback 对“det 框多但识别行少”的情况增加强文本豁免，减少高置信结果继续跑完整增强链路。
  - `VECTRAPARSE_OCR_TRACE=1` 现在会输出 det pass 开始、每 16 个 bbox 的处理进度和 pass 完成信息，用于定位复杂截图耗时阶段。
  - 若拆分结果少于两条、有效字符太少、相比整框识别明显丢内容或置信度下降过多，则保留原整框识别结果。
  - `line-crops` fallback 现在也复用真实行 bbox，不再用行序号估算 y 坐标。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo build --release -p vectraparse-ffi`
  - `gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -L/tmp/onnxruntime-linux-x64-1.26.0/lib -lonnxruntime -ldl -lpthread -lm -Wl,-rpath,/tmp/onnxruntime-linux-x64-1.26.0/lib -o target/extract-static`
  - 使用用户提供复杂截图样本执行 `target/extract-static` 端到端回测
- 真实截图验证结果：
  - 用户提供复杂截图样本中的相邻聊天文本已从合并行拆成两行。
  - 本地完整运行约 102 秒，说明 1920x1032 复杂截图仍有明显耗时，后续需要继续做 det/rec 预算和布局裁剪优化。
- 已完成浅色底黑字主动补充识别：
  - 正常 det 完成后会用小预算尝试颜色区域候选识别，用于补足 det 没框住的浅灰/浅蓝 UI 色块黑字。
  - 颜色区域量化从 3-bit 提升到 4-bit，减少浅色面板和页面背景被归为同一颜色桶导致候选缺失的问题。
  - 区域二值化的背景估计从外框颜色改为区域主色，避免 bbox 外扩后边框落到色块外部，导致整个浅色面板被误判为前景并跳过。
  - 主动补充合并前会过滤与已有 det line 明显重叠的候选；低质量 fallback 仍保留完整颜色区域候选，避免弱结果失去兜底。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
  - `git diff --check -- crates/vectraparse-ocr/src/lib.rs docs/dev/4-plan-ocr-preprocessing-optimization.md`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo build --release -p vectraparse-ffi`
  - `gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -L/tmp/onnxruntime-linux-x64-1.26.0/lib -lonnxruntime -ldl -lpthread -lm -Wl,-rpath,/tmp/onnxruntime-linux-x64-1.26.0/lib -o target/extract-static`
  - 使用用户提供复杂截图样本执行 `target/extract-static` 端到端回测
- 真实截图验证结果：
  - 用户提供复杂截图样本可识别更多页面区域文字，目标合并行仍保持拆成两行。
  - 输出中仍有少量相似内容重复和误字，说明当前优化提升召回但没有解决 rec 模型误识别和语义去重问题；真实截图 golden 仍缺失。
- 已完成 det 候选质量、rec tight crop 和近似重复去重：
  - 原始 det 小框不再直接追加到最终文本；仅在原图 det 主路径中作为合并框的替代切分候选，切分结果需通过 `should_use_split_lines` 内容量/置信度门禁后才替代整框识别。
  - rec 前会基于前景 mask 生成 tight crop，去掉大块空白并保留 padding，降低宽 crop 被压缩后误读的概率。
  - `recognized_from_text_lines` 会在区域聚类前按 bbox 重叠和文本相似度去重；同一列附近的长文本近似重复也会保守去重，短文本不做全局去重。
  - 真实截图回测中，直接追加原始小框会显著增加重复和噪声，已改为“替代候选、不独立追加”的实现。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
  - `rustfmt --edition 2024 --config skip_children=true --check crates/vectraparse-ocr/src/lib.rs`
  - `git diff --check -- crates/vectraparse-ocr/src/lib.rs docs/dev/4-plan-ocr-preprocessing-optimization.md`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo build --release -p vectraparse-ffi`
  - `gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -L/tmp/onnxruntime-linux-x64-1.26.0/lib -lonnxruntime -ldl -lpthread -lm -Wl,-rpath,/tmp/onnxruntime-linux-x64-1.26.0/lib -o target/extract-static`
  - 使用用户提供复杂截图样本执行 `target/extract-static` 端到端回测
- 真实截图验证结果：
  - 用户提供复杂截图样本中的目标合并行仍保持拆成两行。
  - 右侧成员/组织文本的重复比“直接追加原始小框”版本收敛，但仍有少量相近行残留；该类问题后续更适合通过真实 OCR golden 和版面区域先切分继续处理。
  - 本地端到端仍约百秒，说明复杂整页截图的性能预算已成为后续优化约束。
- 已完成短噪声过滤和全局长文本精确去重：
  - 当 OCR 行数较多时，会过滤短 ASCII/符号噪声行，例如孤立大写字母、纯符号和极短非中文片段；短中文名字不会因此被全局删除。
  - 长文本若完全相同或几乎完全相同，可跨位置去重；同列长文本近似重复继续保留较高质量候选。
  - 真实截图回测中，`AM`、`WH`、`Y`、`D`、`+`、`...`、`2+` 等短噪声明显减少，目标聊天合并行仍保持拆分。
- 本轮验证命令：
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-ocr`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo test -p vectraparse-parsers ocr_success_metadata`
  - `ORT_INSTALL_DIR=/tmp/onnxruntime-linux-x64-1.26.0 LD_LIBRARY_PATH=/tmp/onnxruntime-linux-x64-1.26.0/lib cargo build --release -p vectraparse-ffi`
  - `gcc examples/c/extract_static.c -Iinclude target/release/libvectraparse_ffi.a -L/tmp/onnxruntime-linux-x64-1.26.0/lib -lonnxruntime -ldl -lpthread -lm -Wl,-rpath,/tmp/onnxruntime-linux-x64-1.26.0/lib -o target/extract-static`
  - 使用用户提供复杂截图样本执行 `target/extract-static` 端到端回测
- 真实截图验证结果：
  - 用户提供复杂截图样本输出比上一轮少了多处孤立短噪声和一处重复长组织文本。
  - 仍有少量语义相近但 OCR 文本不够相似的组织/成员行残留；继续优化应优先补真实截图 golden，再做页面区域级预切分。
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
  - 当前没有纳入私有真实截图 OCR 文本 fixture；`image/png` 覆盖的是格式元数据，不证明业务截图 OCR 文本识别质量。
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
  - 单测覆盖新增 Otsu、局部二值化、640 动态宽度和候选过滤；仍缺私有真实截图 OCR 文本 fixture。

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
  - 仍缺私有真实截图 OCR 文本 fixture，无法自动判断这些启发式对具体业务样本的净收益。

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
  - 仍缺私有真实截图 OCR 文本 fixture，`color_region_count` 可用于观察该路径是否命中以及候选规模是否失控。

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

## 7.6 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部 crop 切分和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、模型、字典、构建链接或 C ABI。
  - 拆分结果有内容量、置信度和额外 rec 预算门禁；不满足条件时回退到原整框识别；满足条件时不再额外跑整框 rec 比较，避免宽 crop 识别拖慢复杂截图；质量 fallback 也增加强文本豁免，降低误切分或无谓补跑导致丢文本/耗时放大的风险。
- 高级产品：
  - 本轮直接针对“两个视觉消息/区域被合并成一条文本”的用户样本现象：当 det 把不同背景色块、多行或同一 y 行的左右区域包进同一框时，输出可恢复为多条文本。
  - 仍需用户用真实截图验证阈值；复杂背景、行距过小或透明叠层可能仍需要后续调参。
- 高级架构师：
  - 未新增依赖，复用现有颜色距离、Otsu 阈值和 bbox 布局聚类；切分结果仍进入原 `TextLine` 管线，不新增外部 API 负担。
  - 拆行限制在原图 det pass，避免与增强、上采样和旋转 fallback 组合成乘法级成本。
  - 这属于 OCR rec 前的局部 crop 预处理，不改变模型输入输出协议。
- 高级工程师：
  - 单测覆盖两行前景切分、宽行按大空隙切分、相邻背景色块切分和单行不切分；OCR crate 全量测试通过。
  - 当前没有真实截图 golden，无法自动证明用户样本已拆开，需要用户运行 `extract-static` 回测。

## 7.7 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部颜色区域候选和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、模型、字典、构建链接或 C ABI。
  - 主动颜色区域识别有固定小预算；与已有 det line 明显重叠的候选不会主动合并，降低重复噪声风险。
- 高级产品：
  - 本轮针对“浅色背景块 + 黑色字体没识别到”增加主动补充识别，可覆盖 det 未框住但颜色区域明显的 UI 文本。
  - 真实截图回测显示召回增加，但仍存在误字和相似内容重复；后续需要真实 golden 和语义级去重来评估净收益。
- 高级架构师：
  - 未新增依赖，4-bit 颜色量化和区域主色背景估计仍属于轻量图像预处理，不改变 OCR 模型协议。
  - 低质量路径保留完整颜色区域 fallback，正常路径只做保守增量合并，避免把 fallback 逻辑无条件放大。
- 高级工程师：
  - 单测覆盖浅色面板候选检测、带 padding 的浅色面板二值化，以及主动颜色区域跳过已有文本框。
  - 端到端验证重新生成 `target/extract-static` 并回测用户提供复杂截图样本；该样本仍约 102 秒，性能优化仍是后续重点。

## 7.8 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部候选选择、rec 预处理和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - 原始 det 小框只作为替代切分候选，不独立追加到最终文本；真实截图已验证直接追加会放大噪声，因此保留更保守路径。
- 高级产品：
  - 本轮目标是减少复杂截图中“长框压缩误读、相似行重复、合并框串读”的问题，同时保持已有聊天行拆分效果。
  - 回测显示目标聊天合并行仍拆开，右侧重复行有所收敛但未完全消除；剩余问题需要结合真实 golden 做更强的版面分区和语义去重。
- 高级架构师：
  - 未新增依赖，新增逻辑复用现有 bbox、前景 mask 和相似度计算；不改变公开 OCR API。
  - 近似去重只在结构化行进入区域聚类前执行，避免污染底层 det/rec 输出接口。
- 高级工程师：
  - 单测覆盖原始 det 小框替代候选、rec tight crop、bbox 重叠近似去重和同列长文本近似去重。
  - `extract-static` 端到端回测仍约百秒，后续继续提升准确率前应补真实截图 golden 并同步做性能预算。

## 7.9 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部最终行过滤和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - 短噪声过滤只在密集结果中生效，且不删除短中文文本；长文本全局去重要求文本几乎完全相同，避免过度合并真实短名字。
- 高级产品：
  - 本轮针对复杂截图输出中的孤立字母、符号和重复组织名做收口，优先降低可见噪声。
  - 回测显示短噪声明显减少，但语义相近、字面差异较大的组织/成员行仍会残留；后续需要真实 golden 和区域级预切分。
- 高级架构师：
  - 未新增依赖，仍基于已有文本相似度、bbox 和行过滤逻辑，不扩大 OCR 模型调用范围。
  - 该优化属于最终文本清理层，不改变 det/rec 模型输入输出协议。
- 高级工程师：
  - 单测覆盖密集结果短 ASCII/符号噪声过滤、全局长文本精确去重，并保留短中文名字重复。
  - `extract-static` 端到端回测确认目标聊天行仍拆分，短噪声减少；复杂截图耗时仍是残余问题。

## 7.10 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部页面区域候选、bbox 坐标回映和补充候选过滤，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - 新增页面区域 det 有固定上限，区域结果只作为未覆盖强候选补充，避免大范围覆盖基础 det 结果。
- 高级产品：
  - 本轮针对复杂截图中多栏/多面板页面，增加基于检测框横向分布的识别前区域预切分，让局部区域有机会重新 det/rec。
  - 测试用例已移除真实截图中的姓名、组织、编号和原句，只保留合成文本验证通用行为，避免把样本内容固化进动态库。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；区域 crop 通过 `Offset` 坐标回映回原图坐标后复用现有行聚类、去重和补充合并流程。
  - 页面区域识别仍是启发式补充路径，不替代真正的版面分析模型；后续应优先补真实 golden 评估，而不是继续样本定制。
- 高级工程师：
  - 单测覆盖页面列区域候选、crop 坐标回映、区域补充候选过滤以及去重合并。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`cargo check -p vectraparse-ffi`、`rustfmt --check` 和 `git diff --check`；中途用户指出样本特化风险后已清理真实样本文本并重新验证。

## 7.11 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 OCR crate 内部页面区域候选生成和当前任务文档，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - 视觉区域候选仍受 `MAX_PAGE_REGION_DET_PASSES` 限制，并继续走强补充过滤，避免把区域识别结果无条件覆盖基础 det。
- 高级产品：
  - 本轮把页面区域候选从“依赖初始 det 框”扩展为“可基于图像边缘/前景投影生成”，用于复杂多栏截图中初始 det 漏框的场景。
  - 单测只使用合成矩形纹理验证通用视觉分栏，不使用真实截图中的业务文本、姓名、组织或编号。
- 高级架构师：
  - 未新增依赖，视觉分栏基于现有 `image` crate、边缘掩码和列投影；生成的区域仍通过 `Offset` 回映原图坐标并复用现有 OCR 行管线。
  - 该实现仍是轻量启发式布局候选，不替代完整 layout 模型；后续准确率评估应依赖真实 golden，而不是继续固化样本内容。
- 高级工程师：
  - 单测覆盖三列视觉分栏、单列密集内容不切分、以及无 det 框时仍能从视觉候选产生页面区域。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`cargo check -p vectraparse-ffi`、`rustfmt --check` 和 `git diff --check`。

## 7.12 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只增加 OCR trace 结构、JSON 生成和 parser metadata 映射，未触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、构建链接或 C ABI。
  - `VECTRAPARSE_OCR_TRACE_JSON=1` 才生成大 JSON 并输出到 stderr/metadata，默认路径只保留结构化 trace lines，避免默认结果膨胀。
- 高级产品：
  - 本轮补齐准确率优化的可观测基础：每条最终 OCR 行可追踪 `source`、`bbox`、`crop_size`、`confidence` 和 `text`，便于定位错误来自 det、page-region、color-region 或 fallback。
  - 本轮没有新增识别启发式，也没有引入真实截图业务文本作为测试锚点。
- 高级架构师：
  - 未新增依赖，JSON 使用内部转义和拼接；parser 侧仅将 trace line 数量和可选 JSON 写入 metadata。
  - 该 trace 是后续 golden/候选仲裁的输入基础，尚不等同于完整 golden 评估。
- 高级工程师：
  - 单测覆盖 trace line 的 bbox/source/crop_size 和 JSON 文本转义；parser metadata 覆盖 trace line count 与 trace JSON。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`cargo test -p vectraparse-parsers ocr_success_metadata_records_fallback_and_low_confidence`、`cargo check -p vectraparse-ffi`、`rustfmt --check` 和 `git diff --check`。

## 7.13 本轮 Code 阶段审视

- 安全审查员：
  - 本轮新增独立 OCR trace golden 校验脚本和合成 trace fixture，不触碰 OCR 模型、字典、ORT FFI、构建链接或动态库导出接口。
  - golden 样例只使用合成文本 `Alpha/Beta` 和合成 bbox，不包含真实截图业务内容。
- 高级产品：
  - 本轮补齐第 2 步 golden 评估入口：可用 manifest 指向 trace JSON 和期望规则，校验行数、区域数、source、bbox、crop_size、confidence、must-have 和 must-not-have。
  - 该入口先验证 trace 结构和规则机制；真实截图样本需要后续在外部明确授权后再加入 manifest。
- 高级架构师：
  - 未新增 Rust 依赖，脚本基于 Python 标准库解析 JSON；与现有 `scripts/golden_validate.sh` 并行存在，不改变现有 golden 主流程。
  - 规则文件采用 JSON，便于后续扩展候选仲裁、回归阈值和样本分组。
- 高级工程师：
  - 验证已执行 `python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo test -p vectraparse-ocr`、`cargo check -p vectraparse-ffi` 和 `git diff --check`。
  - 当前脚本不运行 OCR 模型，只比较已生成的 trace JSON；后续如要端到端生成 trace，需要单独接入 `extract-static` 或 Rust 测试入口。

## 7.14 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs` 的 OCR 候选、行拆分和颜色区域补充逻辑，不触碰 `ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成 `Alpha/Beta`、矩形色块和通用占位文本，不包含真实截图业务内容。
- 高级产品：
  - 已完成本轮要求的 1/2/3：对空识别或低质/大框行增加局部修复；将带换行的识别结果拆成多条 `OcrTextLine`；颜色区域补充先跳过可靠覆盖区域，再识别未覆盖/可修复重叠区域。
  - 预期改善复杂截图中“整体结果够强但局部仍漏识别或粘行”的场景，尤其是浅色/彩色背景上的前景文字和跨行检测框。
- 高级架构师：
  - 行级修复复用现有 split、二值化和 rec 管线，只增加受限预算 `MAX_LINE_REPAIR_RECOGNITIONS_PER_PASS`，避免无界增加复杂截图耗时。
  - 颜色区域补充从“先识别前 N 个再过滤”调整为“先按可靠文本覆盖过滤再识别”，在不扩大默认识别上限的情况下提高未覆盖色块的命中概率。
- 高级工程师：
  - 单测覆盖换行候选拆成独立 line、低质/大弱框触发修复判断、以及颜色区域覆盖过滤允许修复低质重叠框。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs` 和 `git diff --check`。

## 7.15 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试继续使用合成 `Alpha/Beta` 和矩形纹理，不包含真实截图里的姓名、组织、编号、原句或文件名，避免样本特化。
- 高级产品：
  - page-region 识别现在拥有独立的小型拆行/修复预算，复杂截图中局部区域被重新 det 后仍可继续修复低质大框或空行。
  - 大图在当前识别为空、低置信、det 框多但有效行少，或已有行可修复时，会按纹理密度选择有限 tile 以原图分辨率重新 det/rec，用于补救整页缩放导致的小字漏检。
- 高级架构师：
  - 高分辨率 tile 补充受 `MAX_HIGH_RES_TILE_DET_PASSES` 限制，tile source 不叠加额外拆行/修复预算，避免与 page-region、颜色区域和质量 fallback 形成无界乘法成本。
  - tile 结果仍走现有 `Offset` 坐标回映、行去重、区域聚类和补充过滤流程，不新增公开 OCR API 或外部依赖。
- 高级工程师：
  - 单测覆盖 page-region 本地预算、高分辨率 tile 触发条件、tile 覆盖尾部、纹理 tile 数量上限和阅读顺序。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。

## 7.16 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成 `Alpha/Beta` 和矩形纹理，不包含真实截图中的姓名、组织、编号、原句或文件名，避免样本特化。
- 高级产品：
  - 新增未覆盖纹理区域补识别，用于“整体 OCR 已有结果但局部前景文字未被可靠行覆盖”的复杂截图场景。
  - 视觉 page-region 从列投影扩展到二维面板候选，能覆盖 2x2 卡片/面板页面；候选合并优先走结构化行级去重，减少补充路径带来的重复行。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；visual-region 使用现有 foreground mask、二值化、rec 和 bbox 行聚类管线。
  - visual-region 识别受 `MAX_EAGER_VISUAL_REGION_RECOGNITIONS` 限制，二维 page-region 仍受 `MAX_PAGE_REGION_DET_PASSES` 限制，避免复杂截图成本无界增长。
- 高级工程师：
  - 单测覆盖二维视觉面板候选、未覆盖纹理区域跳过可靠覆盖行，以及既有行级合并去重路径。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。

## 7.17 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成矩形纹理、`Alpha/Beta` 和通用中文占位词，不包含真实截图中的姓名、组织、编号、原句或文件名，避免样本特化。
- 高级产品：
  - 宽行分段 rec 针对“长文本被压缩到 rec 宽度后识别质量下降”的场景，在已有 direct 结果需要修复时才触发。
  - 分段只在存在可靠低前景切点时执行，连续无切点文本不会硬切；分段识别结果必须优于 direct 候选才会采纳。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；宽行分段复用现有 foreground mask、二值化、rec、候选质量和 bbox 行管线。
  - 分段数量受 `MAX_WIDE_LINE_SEGMENTS_PER_LINE` 和已有 repair budget 双重限制，避免长行场景出现无界 rec 调用。
- 高级工程师：
  - 单测覆盖窄空隙长行可分段、连续无可靠切点长行不硬切、ASCII/CJK 分段文本拼接规则。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。

## 7.18 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成色块、矩形纹理和 `Alpha/Beta` 通用占位文本，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 颜色背景区域 direct rec 失败时，局部 det 可在背景块内重新找多行/小字，覆盖“背景色 + 前景文字”被整块 rec 漏掉的场景。
  - 单行/按钮式背景块如果已有可靠行会跳过局部 det；局部 det 输出只作为强 supplement 行合并，降低重复和噪声。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；颜色区域局部 det 复用现有 color-region、det/rec、Offset bbox 映射、行级去重和候选采纳管线。
  - 局部 det 受 `MAX_EAGER_COLOR_REGION_DET_PASSES` 限制，且候选必须存在前景信号，避免无文字装饰色块触发无界 det。
- 高级工程师：
  - 单测覆盖大背景面板部分覆盖仍可作为局部 det 候选、单行已覆盖面板跳过，以及 color-region-det 独立拆行/修复预算。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。

## 7.19 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成灰底和矩形深灰文字，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 低对比深色文字在全局色差低于原阈值时可通过局部均值二值化提取，覆盖浅灰底/浅色背景上的弱对比文字。
  - 纯色低对比区域仍会被拒绝，避免把无文字背景块送入 rec 造成噪声。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；低对比兜底复用现有 `local_binary_luma` 和 `contrast_stretch_luma`。
  - 兜底只在颜色区域二值化原路径失败时触发，不改变强色差路径的优先级。
- 高级工程师：
  - 单测覆盖低对比深色文字可二值化和纯色低对比区域拒绝。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。

## 7.20 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成灰底、矩形纹理和通用占位，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 超大检测框会在 direct rec 后尝试结构化拆行/拆列，减少整块面板或多行内容被 rec 串成一行的机会。
  - tile 重新 det 路径现在也有有限拆分/修复预算，避免高分辨率 tile 内的大面板框只能整块识别。
  - 低对比局部前景 mask 已用于拆行、拆列、tight crop、tile 评分和颜色区域 det 候选评分，弱对比文字不只在最终二值化阶段才有机会被识别。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；新逻辑复用现有前景 mask、行/列投影、候选质量、budget 和 bbox 管线。
  - 强制拆分只有在超大框上触发，且拆分结果必须通过内容/置信门禁，或 direct 结果本身需要修复且拆分文本质量明显更高时才替换。
- 高级工程师：
  - 单测覆盖 tile source 预算、低对比前景 mask 拆行、超大低对比面板结构化拆分和常规单行框不触发强制拆分。
  - 验证已执行 `cargo test -p vectraparse-ocr`、真实复杂截图实测、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。
  - 真实复杂截图实测可正常返回，输出未见明显退化；该样本仍然耗时较高，后续如继续优化应优先做候选成本排序或更细的预算门禁。

## 7.21 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成色块、合成检测框和通用英文占位文本，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 颜色区域、未覆盖视觉区域和颜色区域局部 det 候选现在按文字纹理优先级排序，预算更优先消耗在高收益候选。
  - det 合并会避开足够宽的低纹理沟槽或竖向视觉分隔，减少复杂 UI 面板/卡片之间误合并。
  - 近重复候选不再只看单条最高分，多路径一致文本可通过投票压过孤立高置信变体。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；候选排序、视觉分隔和投票仲裁都复用现有 bbox、foreground mask、文本相似度和质量评分管线。
  - 视觉分隔保护只作用于 det 后处理合并阶段；最终输出仍走现有区域聚类和阅读顺序。
- 高级工程师：
  - 单测覆盖补充候选优先排序、det 合并保留视觉沟槽、近重复多来源投票，以及原有高置信近重复变体选择不回退。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。
  - 本轮未执行无 timeout 的真实复杂截图长跑；后续真实样本验证应使用超时命令，避免复杂补充路径造成不可控耗时。

## 7.22 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs`、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成 panel、矩形前景和 `Alpha/Beta` 等通用占位文本，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 多路径候选会进入全局候选池统一仲裁，减少单一路径补充行因单独评分不够而丢失的情况。
  - 颜色/视觉背景 panel 内可按前景行投影提取分层文字候选，panel 中存在子背景块时可继续递归一层补识别。
  - 局部 crop 识别现在可尝试前景二值化和多窗口局部二值化，颜色区域局部 det 可对 panel crop 做受限局部放大后重新 det。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；候选池复用现有 `RecognizedText`/line 聚类/投票/region 输出，分层区域复用颜色区域、视觉区域、前景 mask 和候选评分。
  - 新增局部放大 det 只作用于受预算限制的颜色背景区域 det 补充路径，bbox 通过内部 `ScaleOffset` 映射回原图。
- 高级工程师：
  - 单测覆盖候选池合并分散行、候选池采纳补充行、分层 panel 前景行提取、局部多窗口二值化候选、局部放大 det 候选顺序和 `ScaleOffset` 坐标映射。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。
  - 本轮仍未新增私有真实截图 OCR 文本 golden；真实复杂截图上的净收益和耗时变化需要用户样本继续回测。

## 7.23 本轮 Code 阶段审视

- 安全审查员：
  - 本轮修改 `crates/vectraparse-ocr/src/lib.rs`、`scripts/ocr_trace_golden.py`、合成 golden expected、当前任务文档和 `docs/dev/README.md` 索引，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试只使用合成线条、合成分隔线和 `Alpha/Beta/Status` 等通用占位文本，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - OCR golden 入口现在能校验最终文本，不只校验 trace 行规则；后续真实截图可通过授权样本进入 manifest。
  - CTC margin、来源权重、多来源支持和视觉边界共同参与候选/版面决策，减少纯字符数或平均置信度驱动的误采纳。
  - 轻量 deskew 和自适应局部预算分别覆盖小角度倾斜和预算噪声问题，不替换模型文件。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR API；CTC margin 是内部候选评分字段，deskew 仅作为整图 fallback，任意角度 bbox 不做伪回映。
  - 视觉边界以可选图像上下文接入区域聚类；无图上下文路径保持原有几何聚类行为。
- 高级工程师：
  - 单测覆盖 CTC margin、基于 margin 的模型候选选择、自适应局部预算、视觉分隔阻止 region 合并、deskew 角度估计和旋转画布扩展。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。
  - 真实私有截图文本 fixture 仍未纳入仓库，后续质量回归仍需用户样本实测或授权 golden。

## 7.24 本轮 Code 阶段审视

- 安全审查员：
  - 本轮只修改 `crates/vectraparse-ocr/src/lib.rs` 和 OCR 计划/索引文档，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试使用 `Alpha/Beta/Invoice/Project` 等合成占位文本和合成纹理，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 行级 margin 现在参与近重复投票和密集结果过滤，减少单个低质行混入高均值结果。
  - 超大框会先尝试结构化拆分，连续宽行在无可靠切点时有重叠滑窗修复路径，降低大框压缩造成的识别损失。
  - 有图像上下文时按视觉 panel 分桶聚类，降低复杂截图左栏、主区、右栏混在同一输出块的概率。
- 高级架构师：
  - 未新增依赖，不改变公开 OCR 输出结构；新增 margin 字段只存在于内部 `TextLine`，CTC 小 beam 仅作为 greedy 的保守备选。
  - 颜色层级增强复用已有前景 mask 和颜色 panel 递归，只增加前景组件候选，不引入业务词典或语义纠错。
- 高级工程师：
  - 单测覆盖行级 margin 投票、低质 ASCII 噪声、超大框优先拆分、结构化拆分门禁、宽行滑窗、滑窗重叠文本合并、panel 归属、颜色前景组件和 CTC beam 备选。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和样例文本禁入检查。
  - 真实复杂截图仍未作为 golden 固化，阈值收益和误伤风险需要用户样本继续回测。

## 7.25 本轮 Code 阶段审视

- 安全审查员：
  - 本轮修改 `crates/vectraparse-ocr/src/lib.rs`、`crates/vectraparse-parsers/src/lib.rs` 的合成测试构造、`scripts/ocr_trace_golden.py`、合成 OCR golden 和 OCR 计划/索引文档，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、FFI ABI 或构建链接。
  - 新增测试和 golden 只使用 `Alpha/Beta/Project/Header` 等通用占位文本和合成 mask/纹理，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - CTC beam 从路径截断升级为 prefix 概率合并，对 blank、重复字符和相同 collapse 文本的概率聚合更稳。
  - det map 连通域会按原始 mask 轮廓投影细化，多行被 dilation 粘成大框时更容易拆回多个候选框。
  - trace 和 OCR golden 现在可观察 line margin、平均/最低 margin、低 margin 行数，外部私有样本可直接用指标规则做回归。
- 高级架构师：
  - 本轮未新增依赖；公开 OCR line/trace 结构增加 `avg_margin`/`min_margin` 字段，C ABI 未变化。
  - 版面聚类从逐行贪心改为文本行图连通分量，再复用既有视觉分隔和 region merge 门禁；仍是启发式版面，不做语义理解。
  - glyph textness 和结构化候选评分复用现有前景 mask、source family、margin 和 readable ratio，不引入业务词典或真实样本规则。
- 高级工程师：
  - 单测覆盖 det 轮廓投影拆分、trace margin JSON、文本行图聚类、glyph textness、CTC prefix beam 概率合并和原有 OCR 辅助路径回归。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、parser OCR metadata 定向测试、`cargo check -p vectraparse-ffi` 和 `rustfmt --edition 2024 --config skip_children=true crates/vectraparse-ocr/src/lib.rs`。
  - 真实截图 OCR 文本质量仍需要用户样本回测；本轮改动提升泛化机制和可观测性，不声明已覆盖未授权私有样本。

## 7.26 本轮 Code 阶段审视

- 安全审查员：
  - 本轮修改 `crates/vectraparse-ocr/src/lib.rs`、`crates/vectraparse-parsers/src/lib.rs` 的合成测试构造、`scripts/ocr_trace_golden.py`、合成 OCR golden 和 OCR 计划/索引文档，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、C ABI 或构建链接。
  - 新增测试和 golden 只使用 `Alpha/Beta` 等通用占位文本、合成色块和合成检测框，不包含真实截图中的姓名、组织、编号、原句或文件名。
- 高级产品：
  - 主 det 结果在原图坐标下可直接使用图像上下文版面分组，复杂页面中的 panel/gutter 边界更早参与最终输出聚类。
  - trace 现在记录候选事件、采用/拒绝原因、候选分数和 candidate_count，便于区分漏检、候选被过滤和候选仲裁失败。
  - 空结果会更早进入视觉前景补扫；dominant 背景软前景和更保守 det 合并分别改善浅底深色字漏召回和相邻 UI 块串读。
  - 长行动态 rec 宽度回到 960 上限，超长行可使用更多分段；CTC confidence/margin 统一按帧级概率校准。
- 高级架构师：
  - 公开 OCR trace Rust 结构增加 `OcrTraceCandidate` 和 `OcrTrace.candidates`，parser metadata 增加 `image.ocr.trace_candidate_count`；C ABI、模型和 ORT FFI 不变。
  - 图像上下文只用于 `BboxTransform::Identity` 的 det 结果；增强、上采样、旋转和局部 det 的回映坐标仍走几何聚类，避免图像坐标系错配。
  - dominant 背景软前景仍受前景比例和 glyph textness 门禁限制，不引入业务词典或真实样本规则。
- 高级工程师：
  - 单测覆盖候选 trace JSON、候选拒绝原因、空结果 visual supplement、小 panel gap 不合并、dominant 软前景、动态宽行分段、960 rec 宽度和 CTC 概率校准。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、parser OCR metadata 定向测试、`cargo check -p vectraparse-ffi`、`rustfmt --edition 2024 --check --config skip_children=true crates/vectraparse-ocr/src/lib.rs`、`git diff --check` 和真实样本文本禁入检查。
  - 本轮仍未加入真实截图文本 golden，复杂页面实际准确度和耗时需要用户继续用私有样本回测。

## 7.27 本轮 Code 阶段审视

- 安全审查员：
  - 本轮继续只修改 `crates/vectraparse-ocr/src/lib.rs`、`crates/vectraparse-parsers/src/lib.rs`、`scripts/ocr_trace_golden.py`、合成 OCR golden 和 OCR 计划/索引文档，不触碰 `crates/vectraparse-ocr/src/ort.rs`、ONNX 模型、字典、C ABI 或构建链接。
  - 新增回归指标只包含 `support_count`、`readable_ratio`、`char_min_confidence`、candidate action/source 计数等泛化字段，不写入真实截图文本、业务词或私有路径。
- 高级产品：
  - 候选池由“更长文本优先”继续收紧为“近重复行支持度 + 行级质量”仲裁，降低多路径补识别把噪声长文本带入最终结果的概率。
  - page-region 识别增加递归 panel-first 子区域展开和局部低阈值 det，有利于复杂截图中的嵌套色块、小面板和弱 det 区域。
  - 多前景 mask 融合和字符级最小置信度 trace 提升了浅底深色字、弱对比行和私有样本回归诊断的可观测性。
- 高级架构师：
  - 未新增依赖；递归 panel 只展开到受限深度和数量上限，低阈值 det 只在局部弱结果 panel 上触发，仍保持预算可控。
  - 公开 Rust OCR line/trace 结构增加 `char_min_confidence`、`readable_ratio` 和 `support_count`；parser metadata 增加 adopted/rejected/source 计数；C ABI 和 FFI 不变。
  - 私有样本回归入口仍基于 trace/golden 指标，不在仓库中固化真实文本 fixture。
- 高级工程师：
  - 单测新增覆盖重复候选 `support_count`、多前景 mask 变体、panel 子区域候选、低阈值 det 门限，以及扩展后的 trace JSON/metadata 字段。
  - 验证已执行 `cargo test -p vectraparse-ocr`、`python3 scripts/ocr_trace_golden.py tests/golden/ocr/manifest.tsv`、parser OCR metadata 定向测试、`cargo check -p vectraparse-ffi` 和 `git diff --check`；`vectraparse-parsers` 整文件 `rustfmt --check` 仍会触发历史格式差异，因此本轮只对 OCR crate 做定向格式检查。
  - 真实复杂截图的净收益和耗时仍需用户私有样本回测；本轮主要补足行级仲裁、递归 panel 和指标回归闭环。

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
| 2026-06-01 | 部分完成执行计划步骤 7：处理长行与英文行 | 已完成逐 crop 中英文结果选择和动态宽度；后续补入宽行分段 rec，但真实长英文 OCR fixture 仍未落地 |
| 2026-06-01 | 完成执行计划步骤 8：增加旋转保守兜底 | 已完成 90/180/270 度旋转图重新 det+逐框 rec 和整图识别兜底；后续步骤 27 已补入轻量整图 deskew fallback |
| 2026-06-01 | 完成执行计划步骤 9：增强 OLE 嵌入图片 OCR 候选处理 | 通过 `.doc` 候选去重、预算告警和失败诊断降低嵌入图 OCR 对主提取链路的干扰 |
| 2026-06-01 | 部分完成执行计划步骤 10：收口 golden/文档 | 已覆盖 `image/png` parser 元数据；步骤 27 后已有 OCR trace/text 规则入口，私有真实截图文本 golden 尚未落地 |
| 2026-06-01 | 修复 PNG OCR 回归：统一 rec 白底预处理并移除 det 无框伪整图框 | 针对普通 RGBA PNG 无法识别的回归，恢复 det/rec 图像前处理一致性并让整图 fallback 只在真实 fallback 阶段触发 |
| 2026-06-01 | 按当前代码实现校准 OCR 计划文档 | 移除与代码不一致的 960 rec 宽度、真实 OCR golden、旋转逐框识别等完成表述 |
| 2026-06-01 | 完成本轮 OCR 识别质量优化 | 放开长行 rec 宽度到 960，增加 Otsu/局部二值化增强变体，并增加行级候选质量过滤 |
| 2026-06-01 | 完成 OCR 质量补强优化 2/3/5/6 | 对增强/上采样/旋转变体重跑 det，增加部分成功补跑合并、det NMS/unclip 扩边和透明图黑底增强 |
| 2026-06-01 | 完成区域级布局聚类 | 将复杂截图 OCR 输出从全局 y/x 排序升级为区域聚类输出，并增加区域诊断 metadata |
| 2026-06-01 | 完成颜色背景区域 fallback | 针对明显背景色块上的文字增加色块候选、区域二值化识别和颜色区域诊断 metadata |
| 2026-06-01 | 完成结构化 OCR regions/lines 和 trace metadata | 让复杂截图 OCR 结果可按区域、行、来源和 fallback 路径调试 |
| 2026-06-01 | 完成 det 框内部多行/多段切分 | 避免单个检测框覆盖相邻消息行或跨左右区域时被 rec 模型粘成一条文本 |
| 2026-06-01 | 收紧复杂截图 OCR 成本 | 将动态 rec 宽度上限调整为 640，并限制大 crop 增强、子段识别和 trace 进度输出 |
| 2026-06-01 | 增加聊天 UI 时间标记断行 | 覆盖左侧会话预览时间与右侧气泡内容被同一 OCR 行串读的样本 |
| 2026-06-01 | 增加浅色底黑字主动补充识别 | 用 4-bit 颜色区域候选、区域主色二值化和重叠过滤提升 UI 色块黑字召回 |
| 2026-06-01 | 增加 det 候选质量、rec tight crop 和近似重复去重 | 降低合并框误读、宽 crop 空白干扰和复杂截图重复噪声 |
| 2026-06-01 | 增加短噪声过滤和全局长文本精确去重 | 减少复杂截图中的孤立字母、符号和重复长组织文本 |
| 2026-06-01 | 增加页面区域预切分补充识别 | 基于检测框横向分布对复杂截图做受限区域 det/rec，并清理测试中的真实样本文本以保持泛化约束 |
| 2026-06-01 | 增加视觉布局页面区域候选 | 通过图像边缘/前景列投影生成多栏页面区域，降低初始 det 框漏检时的召回依赖 |
| 2026-06-01 | 增加 OCR trace JSON | 输出行级 source/bbox/crop_size/confidence/text 诊断信息，为后续 golden 和候选仲裁提供依据 |
| 2026-06-01 | 增加 OCR trace golden 入口 | 用 manifest + expected JSON 校验 trace 行级 source、bbox、crop_size、confidence 和 must-have/must-not-have 规则 |
| 2026-06-01 | 增强 OCR 行级修复与颜色区域补充 | 对低质/大框行做局部二次识别，把换行识别结果拆成独立 line，并优先识别未被可靠文本覆盖的颜色背景区域 |
| 2026-06-01 | 增加 page-region 修复和高分辨率 tile 补充 | 为复杂截图小字、局部漏框和大图缩放后漏检场景增加受限区域内修复与原图 tile 重新 det |
| 2026-06-02 | 增加未覆盖纹理区域和二维区域补充 | 为复杂截图中可靠 OCR 行之外的漏识别文字、2D 面板布局和补充候选重复合并问题增加通用处理 |
| 2026-06-02 | 增加宽行分段 rec | 对低质超宽文本框按可靠低前景切点分段识别并拼接，减少长行压缩对 rec 的影响 |
| 2026-06-02 | 增加颜色背景区域局部 det 补识别 | 对有前景信号的颜色背景块做受限 crop det/rec，补充整块 rec 漏掉的多行和小字 |
| 2026-06-02 | 增加低对比局部二值化兜底 | 在颜色区域全局色差不足时用局部阈值提取深色文字，同时拒绝纯色低对比块 |
| 2026-06-02 | 增加大框强制拆分、tile 本地预算和低对比 mask 拆行拆列 | 针对复杂截图中超大检测框、tile 内大面板框和弱对比文字拆分前漏检继续提升召回 |
| 2026-06-02 | 增加补充候选排序、det 合并视觉分隔和近重复投票 | 优先识别高收益候选，降低跨面板误合并，并让多路径一致文本在仲裁中胜出 |
| 2026-06-02 | 增加全局候选池、分层背景区域和局部多预处理/多尺度 det | 让多路径候选统一仲裁，panel 内文字按前景行递归补识别，并提升弱对比/小字 crop 召回 |
| 2026-06-02 | 增加文本 golden、CTC margin、视觉版面边界、deskew 和自适应预算 | 一次性完成后续准确性优化 1-7，在不替换模型的前提下提升候选选择和版面决策稳定性 |
| 2026-06-02 | 增加行级 margin、超大框优先拆分、宽行滑窗、panel 分桶、颜色组件、噪声过滤和 CTC beam | 一次性完成新一轮准确性优化 1-7，在不替换模型的前提下继续提升复杂截图的行级准确率和版面稳定性 |
| 2026-06-02 | 增加 CTC prefix beam、det 轮廓投影、trace margin、文本行图、glyph textness、候选评分校准和 OCR trace 指标规则 | 一次性完成本轮准确性优化 1-7，以泛化机制和可观测指标继续提升复杂截图 OCR 稳定性 |
| 2026-06-02 | 增加主 det 图像上下文版面、候选 trace 事件、空结果视觉补扫、dominant 背景软前景、保守 det 合并、长行动态宽度/分段和 CTC 概率校准 | 一次性完成本轮准确性优化 1-7，继续降低复杂页面误合并和漏召回，并提升 trace 可诊断性 |
| 2026-06-02 | 增加候选池行级 support 仲裁、递归 panel-first 区域识别、低阈值局部 det、多前景 mask 融合、字符级 trace 指标和 trace 指标/metadata 回归入口 | 一次性完成本轮准确性优化 1-7，继续降低复杂页面误选和弱 det 漏召回，并补足私有样本回归所需的泛化指标 |
