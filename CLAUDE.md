# VoiceTransl 工程描述（AI 自动加载 · 勿删）

> 每次新会话开始会自动读入本文件。**凡是你这个会话改了代码，结束前必须把改动写进文末【工作日志】顶部并刷新【当前状态】**，
> 这样下一个 AI（无论 Claude/Codex/其他）不用再通读全工程就能接上。同一规范见同目录 `AGENTS.md`（内容一致，供 Codex 等读取）。
> 兜底：Claude Code 会话结束时 hook 会自动在 `.claude/session_history.md` 追加一条 git 机械记录；本文件的【工作日志】请自己补文字摘要。
> 整个工程（含 PixelPlayer、WNovelArchiver）的完整对接词：`C:\Users\12413\codex_sessions_summary\整个工程总对接词.md`

## 这是什么
桌面端（Windows，Python 3.11 + Qt，入口 `app.py`）：
- **A. 日语 ASMR 音频流水线**（成熟，勿重写）：识别 → 断句 → AI 翻译 → 校对 → SRT/LRC 字幕输出。
- **B. 局域网媒体库服务器**（核心新功能）：`media_library.py`（SQLite schema 2，RJ/VJ/BJ 归组 + 整理，只读扫描先行）
  · `lan_service.py`（UDP 49731 配对 / HTTP 49732：串流、下载、同步、整理、metadata-refresh）
  · `dlsite_metadata.py`（DLsite 元数据抓取）。
数据源：`D:\音声`（约 1100 文件 / 127 部作品，**只读**）。手机端对端为 PixelPlayer。

## 当前状态
- 分支 `feat/qwen-streaming-integration`；上游整合、OpenCode Go 与 DeepSeek 官方 V4 Flash（用户称 V4.1）适配均已完成，正在按用户要求提交并推送到 `origin` 同名分支。
- 已添加上游远端 `upstream=https://github.com/shinnpuru/VoiceTransl.git`，并按当前 PySide6/Qwen 模块架构分批整合至上游 `b5f7e50` 的有效更新；禁止用 reset/checkout 覆盖本分支的 Qwen、媒体库、LAN、OpenCode 改动。
- 回归基线：仓库 `.venv` 执行 164 项测试全部通过，1 项依赖完整本地 CrispASR 二进制/模型/音频夹具的集成测试跳过；GitHub Actions 构建 YAML 可正常解析。

## 已完成 / 勿重做
- Qwen UI（Qt，顶部多页签）；断句/校对支持**独立服务商+模型+Key+地址**页签，可“跟随主翻译”；翻译/断句/校对三个思考模式开关。
- AI 全量断句：AI 只返回 `start_id/end_id` 行号区间，原文**本地无损拼接**；AI 分组即最终分组。
- 校对：deepseek 系列非思考更快；整份发送需显式高 `max_tokens`（防 `finish_reason=length` 截断后重试）。
- Token 计数 = **实时已接收字符**（不含 hidden reasoning_content）。
- 进度条 = 已完成文件 x/y + 按 stage 从 0 的分阶段百分比。
- Faster-Whisper/PyTorch **必须在 PyQt 之前预加载**，否则 `c10.dll WinError 1114`。
- **OpenCode Zen**：服务商下拉可直接选择，地址 `https://opencode.ai/zen`；模型发现读取 `/v1/models`。DeepSeek/GLM/MiniMax/Kimi 等走 Chat Completions，GPT/Grok/Muse 走 Responses；Claude/Qwen（Anthropic Messages）与 Gemini 专用端点暂不展示，避免误选不兼容协议。

## 硬性规则
- 不自动 commit/push/建 PR（被明确要求才做）；不调用付费 API、不做真实翻译。
- 不对 `D:\音声` 或手机真实媒体做增删改；文件整理只能先预览、经确认后才执行。
- 保留全部未提交修改；不用 `git reset`/`checkout` 清理工作区。
- 测试只放临时目录并在测试后清理；用中文汇报。

## 工作日志（新在上）
- 2026-09-13 参照 CC Switch 的预设式设计补齐 DeepSeek 官方 V4 Flash（用户称 V4.1）：新增集中式服务商预设，官方模型固定使用 `deepseek-v4-flash`/`deepseek-v4-pro`，兼容常见 V4.1 手输别名并在发送前规范化；DeepSeek 官方与 OpenCode Go 切换时分别一键回填正确模型，避免地址/模型串线。模型获取改用 Qt 异步网络、15 秒硬超时和候选 `/models` 端点，获取期间界面保持响应，结果可直接回填主翻译/断句/校对。全程仅用本地模拟服务验证，未调用付费 API；全套 164 项测试通过、1 项本地 CrispASR 集成测试跳过。
- 2026-09-12 完整适配 OpenCode Go 的 DeepSeek V4.1 Flash：新增独立服务商和 `https://opencode.ai/zen/go` 默认地址，模型 ID 使用官方 `deepseek-flash`；主翻译、AI 断句、校对、模型发现均走 Chat Completions，补齐思考模式识别和 384K 输出上限。使用模拟 HTTP/SDK 响应验证，未调用真实付费 API；全套 158 项测试通过、1 项本地 CrispASR 集成测试跳过。按用户要求提交并推送到 fork 的 `feat/qwen-streaming-integration`。
- 2026-09-12 分批整合 `shinnpuru/main@b5f7e50`：纳入 CI/Release、Windows Vulkan/CUDA/FFmpeg 打包、macOS 目录产物、去 Torch 音频分离、英文 README/图标、上游测试集和 Windows UTF-8 日志；按本分支模块架构移植 FFmpeg 路径解析、CrispASR VAD/token 参数、离线 ASR/翻译模型自检、llama-server 命令校验、在线/本地翻译器分流及即时界面语言切换。旧版 PyQt5 单文件 UI 补丁未直接覆盖现有 PySide6/Qwen/媒体库/OpenCode 实现。全套 154 项测试通过、1 项本地模型集成测试跳过；逐步提交到本地功能分支，未 push。
- 2026-09-12 将既有工作区改动分逻辑落袋：`3a1e55e` 提交桌面媒体库、DLsite 元数据、LAN 同步和测试；`1777037` 提交 OpenCode Zen 双协议接入和测试。仓库 `.venv` 全套测试退出码 0；均只提交到本地功能分支，未 push。
- 2026-09-11 新增 OpenCode Zen API：加入服务商映射、在线模型发现过滤、Chat Completions + Responses 双协议路由，覆盖主翻译、独立 AI 断句/校对和 Token 可用性检查；新增 Responses 流式文本/完成原因适配及清晰的不兼容模型报错。全程使用模拟响应，未调用真实付费 API；`python -m unittest discover -s tests -p 'test_*.py'` 66/66 通过，未 commit/push。
- 2026-09-06 建立 CLAUDE.md/AGENTS.md 工程描述 + 工作日志约定；核对本仓库与 PixelPlayer、WNovelArchiver 状态并归档到 `C:\Users\12413\codex_sessions_summary\`。
