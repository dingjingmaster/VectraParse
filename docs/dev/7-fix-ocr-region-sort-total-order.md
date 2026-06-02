# OCR 区域排序全序修复

> 文档元数据
> - 文件编号：7
> - 文档类型：fix
> - 文件路径：docs/dev/7-fix-ocr-region-sort-total-order.md
> - 文档版本：v1.0.0
> - 最后更新：2026-06-02
> - 需求级别：L2
> - 关联需求：修复 `extract-static` 在复杂截图上因排序比较器非全序导致的运行时 panic

## 1. 问题

- 现象：
  - 运行 `./target/extract-static /home/dingjing/files/b.png` 时触发
    `user-provided comparison function does not correctly implement a total order`。
- 影响：
  - OCR 结果无法返回，进程在区域排序阶段 panic。

## 2. 根因

- `crates/vectraparse-ocr/src/lib.rs` 中 `reading_region_order` 采用 pairwise 动态规则决定先比 `x` 还是 `y`。
- 该比较器不满足传递性，在复杂布局下会形成排序环，触发 Rust 排序实现的运行时保护。

## 3. 修复方案

- 将区域排序改为基于固定 key 的严格全序：
  - 先按量化后的 `center_y` bucket
  - 再按 `x`
  - 再按 `y / bbox` 兜底
- 补单测覆盖比较器全序特性。

## 4. 验证

- `cargo test -p vectraparse-ocr reading_region_order_is_total_for_staggered_layouts`
- `cargo test -p vectraparse-ocr`
- `./target/extract-static /home/dingjing/files/b.png`
  - 已确认不再出现 `user-provided comparison function does not correctly implement a total order` panic
