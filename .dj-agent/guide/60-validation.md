# 60-validation.md

> 导航：上一节 `50-review-checklist.md` ｜ 返回 `00-routing.md`
>
> 验证命令约定：把“验证通过”变成可执行标准。

## 1. 默认策略

- 优先使用项目已有脚本：`Makefile`、`package.json`、`go.mod`、`pyproject.toml`、`Cargo.toml`、`pom.xml` 等。
- 优先级：单元测试 > lint/typecheck > build > 可复现实测步骤。
- 改动越小，验证越窄；影响公共接口、架构或数据时，验证必须扩大。
- Bug 修复优先执行复现用例和回归验证；E2/E3 必须回归验证，不能稳定复现时执行相关验证并记录证据链和残余风险。
- 命中 `05-safety-gates.md` 高风险开发门禁时，验证必须覆盖对应风险项。
- 验证命令失败时，先判断是否由本次修改导致；非本次问题要明确记录。
- L2+ 或命中高风险门禁时，必须记录验证环境、未执行验证项和残余风险。

## 2. 常见项目命令

| 项目类型 | 优先验证命令 |
|---------|-------------|
| C/C++ | 项目既有构建/测试命令、`ctest`、`make test`；涉及内存时优先 ASan/UBSan；涉及并发时优先 TSan；必要时 Valgrind |
| Kernel | 目标模块构建、相关自测、`checkpatch`/`sparse`（如项目使用）；涉及内存/并发时记录 KASAN/KCSAN/lockdep 是否可用 |
| Go | `go test ./...`；涉及并发时 `go test -race ./...` |
| Node.js | `npm test`、`npm run lint`、`npm run build`、`npm run typecheck` |
| Python | `pytest`、`ruff check .`、`mypy .` |
| Rust | `cargo test`、`cargo clippy`、`cargo build`、`cargo fmt --check`；涉及 `unsafe` 时增加边界、错误路径和 FFI 验证 |
| Java/Maven | `mvn test` |
| Java/Gradle | `./gradlew test` |
| 文档仓库 | 链接检查、`git diff --check`、人工阅读一致性检查 |

## 3. 分级验证要求

| 级别 | 最小验证 |
|------|---------|
| L0 | 代码阅读或单个针对性命令 |
| L1 | 复现/验证目标问题，必要时跑相关测试；E2/E3 bug 修复必须有回归验证，E1 记录相关验证、证据链和残余风险 |
| L2 | 执行计划中的验证项，至少跑相关自动化或说明无自动化原因；覆盖命中的安全门禁风险 |
| L3 | 相关模块完整测试 + 构建/类型/静态检查；底层/并发/内存相关改动优先增加 sanitizer/race/静态检查 |
| L4 | 全量验证 + 部署/回滚/监控检查；必须记录回滚和止损方案 |

## 4. 无自动化验证时

记录以下内容：
- 验证环境：本地/容器/测试环境。
- 系统信息：OS、内核版本、架构、编译器/运行时版本（适用时）。
- 操作步骤：可由他人复现。
- 预期结果：明确成功标准。
- 实际结果：说明是否通过。
- 未执行项：未运行的关键验证及原因。
- 风险：无法覆盖的场景。
