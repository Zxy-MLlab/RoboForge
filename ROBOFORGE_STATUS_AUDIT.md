# RoboForge 项目状态审计

审计日期：2026-09-01  
审计范围：`/root/autodl-tmp/RoboForge`（用户入口 `/root/autodl-pub` 为其映射路径）  
审计方式：只读检查当前代码、未提交 diff、OpenHands 持久化状态、实验 evidence、进程和回归测试。本审计没有启动新的长时间实验，也没有修改代码。

## 1. 目标重新确认

RoboForge 的最终目标是一个面向具身智能的 Embodied Coding Harness：强 LLM 像 Codex 一样理解未知机器人任务，观察环境，读写 Controller，执行真实 physical trial，检查 RGB/RGB-D/轨迹/本体感觉和验证结果，分析失败，修改代码，检索并复用 Experience/Skill/Capability，在必要时获取新能力，并把有效经验持久化到后续未知任务。

当前正式运行路径仍围绕该目标：`Frontier LLM -> OpenHands Agent/LocalConversation -> RoboForge embodied tools -> authenticated Unix RPC -> Python 3.11 Adapter -> LIBERO`。没有发现新 `roboforge/` 运行时重新实现通用 AgentLoop、Workspace、Provider 或编辑器。目标没有发生架构性偏离；但“自主获取全新 Capability 的真实闭环”仍未被证实，因此整体目标尚未完成。

## 2. 当前架构审计

### Agent Layer

- **Framework：完成。** 正式入口 `roboforge-openhands` 使用 OpenHands SDK `Agent` 和 `LocalConversation`（doctor 曾验证 SDK 1.44.1）。
- **OpenHands/Codex/自研 loop：完成。** OpenHands 负责 LLM 交互、事件历史、上下文、重试、workspace、文件编辑、terminal 和 session lifecycle；RoboForge 只注册 embodied tools。没有第二套 generic AgentLoop 作为正式运行核心。
- **规划：部分完成。** Agent 可以规划并执行多轮工具调用；但 OpenHands 对“普通文本响应”默认标记本轮 `FINISHED`，当前 RoboForge 有 embodied continuation 适配。模型仍可能在诊断后不调用下一工具。
- **代码修改：完成。** Agent 可通过 OpenHands file editor/terminal 修改 workspace 内 `controller.py` 和新的 Python capability 文件，编辑边界受限。
- **经验调用：部分完成。** 搜索、读取、物化、资产 provenance 均可由 Agent 调用；真实轨迹证明了已有 Experience/Capability 的检索和使用，但不是每条失败轨迹都能正确先读后用。

### Experiment Runtime

- **Physical trial：完成。** `run_controller` 通过 ExperimentService、RPC、Adapter 执行；每次提交形成 physical experiment，受 trial budget 约束。
- **Diagnostic：完成。** `observe`、`inspect_trial`、`compare_trials` 为只读诊断，不消费 physical budget；已有真实 diagnostic evidence。
- **Reset：完成。** Adapter 在真实 S0 reset 后捕获 canonical before view；此前 before-frame lifecycle bug 已修复并保留修复证据。
- **Evidence：完成。** Controller snapshot/digest、before/after RGB、RGB-D、proprioception、action trace、verifier、artifact hash 和 provenance 持久化到不可变 CAS/ledger。
- **Artifact：完成。** 图片、深度、JSON trace、视频及 URI/hash 可检查，且与 experiment 绑定。
- **Failure recovery：部分完成。** durable resume、pending reservation、stale socket 修复和 fail-closed idempotency 已实现并测试；旧 Task B 曾记录一次 recovery incident，说明真实 crash/replay 边界仍需持续演练。

### Memory / Experience

| 能力 | 状态 | 证据与缺口 |
|---|---|---|
| Experience memory | 部分完成 | `experience://96b240...` 已持久化并在多条真实轨迹 `assets_used` 中复用；成功/失败摘要和适用性自动蒸馏仍不完整。 |
| Skill memory | 部分完成 | 资产模型和持久化目录存在，但没有足够真实证据证明 Agent 自主蒸馏、检索、适配 Skill 的完整闭环。 |
| Capability memory | 部分完成 | 已有 `capability://072169...`，含 source digest、验证、读取和物化；缺少真实 Agent 创建全新 Capability 的完整证据。 |
| Controller evolution | 完成（研究级） | Task 2/7/8 等真实轨迹有多次 Controller digest、diff、trial comparison 和物理反馈驱动修改；并非所有轨迹都改善或成功。 |
| Cross-task reuse | 部分完成到完成（有条件） | Task 8/9/F 证明已有 Experience/Capability 的搜索、读取、物化、物理使用和 provenance；尚未证明新能力跨任务产生并稳定迁移。 |

## 3. 阶段性成果

### 阶段 1：Generic Coding Harness

**结论：完成。** 正式入口由 OpenHands SDK 构造 Agent/LocalConversation；静态边界测试确认新 runtime 不依赖旧 generic `AgentLoop`/Workspace/Context Builder；OpenHands 持久化事件和 workspace 正常工作。回归结果为 `279 passed, 2 skipped`，2 个 skip 是基础 Python 没有 `openhands.sdk`，非功能失败。仍需在 OpenHands 可用的 CI 环境中取消这两个 skip。

### 阶段 2：Embodied Experiment Infrastructure

**结论：完成。** 真实 LIBERO 轨迹已经产生 observation、diagnostic、physical trial、RGB/RGB-D、proprioception、视频、action trace、Controller digest 和 authentic verifier receipt。Task 8 的 `experiment://physical-000006` 是真实 `verified=true` 成功；before-frame reset 修复由后续 distinct before/after hashes 证明。RPC token、预算、不可变 evidence、pending reservation 和 stale socket 恢复均有代码/测试/实验记录。仍缺少更系统的长期 crash/restart 压力矩阵。

### 阶段 3：Capability Acquisition

**结论：机制完成，真实自主闭环未完成。** 已有 capability 的 sandbox validation、source digest pinning、register、read、materialize、import 和 physical use；Task 6 是真实物理使用证据，Task 2 记录了 Agent 试图 acquisition 且安全拒绝越界 source。空能力库实验和 Task 9 acquisition 实验没有证明 Agent 创建新 `.py` capability 后成功注册、读取、物化、接入 Controller、物理使用并持久化。该阶段不能标记完成。

### 阶段 4：Embodied Debugging / Self-improvement

**结论：部分完成。** Task 2/7/8 有真实多 trial Controller 演化、对比和失败分析；Agent 可观察实际证据并改变代码。缺口是改善率和自主诊断质量不稳定，失败后常重复策略或停留在诊断，尚无一致的“失败原因 -> 能力缺口 -> 能力获取 -> 改进”闭环。

### 阶段 5：Cross-task Experience Transfer

**结论：部分完成且已有强证据。** Task 8、Task 9、Task F 证明 Agent 可检索/读取既有 Experience 或 Capability，`assets_used` 将其绑定到真实 physical experiment，且可被 Controller 导入使用。Task 8 同时有 authentic success。缺口是新 Capability 的跨任务产生和适配尚未证明，已有 Experience 的迁移也不是每次都带来成功。

## 4. 最近长时间运行分析

最近长运行使用 `gpt-5.6-sol`、Apex、Chat Completions、vision override、continuation 适配，并把预算提高到 50 trials/400 iterations。当前检查显示：无运行中的 `roboforge`/`rpc_server` 进程；最近 durable run `/tmp/rf-resume-final-CLQuE1` 已有 7 个 physical evidence 和 6 个 diagnostic evidence，所有 physical receipt 均 `verified=false`，无 capability 文件或 `assets_used`。

根因分类：

- **A：代码问题，已修复但不是主因。** 曾存在 Responses API 误选、vision 元数据误判、迭代上限未在构造器传递、普通文本终止不适合 embodied、stale socket 和事件导入路径错误；这些均已修复并有 `13 passed` 原生测试或运行证据。
- **B：实验设计问题，仍存在。** 物理 trial 与 Agent iteration 的成本/时长很高；仅增加预算不能保证模型改变策略，空能力库任务也没有构造可观测且必然需要新能力的缺口。
- **C：Agent workflow 问题，主因之一。** 模型在 observe/diagnostic 后常返回文本或重复诊断，不能稳定形成“比较失败 -> 判断能力缺口 -> acquisition”动作序列。continuation 只保证 loop 继续，不保证语义决策。
- **D：模型/Provider 问题，主因之一。** Apex 对 OpenHands 多轮大请求多次出现 `Server disconnected without sending a response` / TLS EOF；短 chat/tool 请求可成功，但长上下文成功率不稳定，导致轨迹被截断。
- **E：目标定义问题，次要。** “自主获取新 Capability”要求很强，但当前 public task 未必需要额外通用软件能力；模型不调用 acquisition 不能直接证明 Harness 缺陷。必须区分合理不获取与能力缺口未识别。

综合判断：最近几小时没有结果不是单一“模型失败”，而是 **D（Provider 长请求不稳定）+ C（workflow 不强制进入能力决策）+ B/E（任务缺口与验收证据设计不足）** 的组合；A 类本地问题已经显著减少。

## 5. 最大三个技术瓶颈

### 瓶颈一：长上下文 Provider 可靠性

**为什么存在：** OpenHands 的真实请求包含系统提示、13 个工具 schema、多模态 evidence 和历史；Apex 的短 `/chat/completions` 可用，但长多轮请求间歇性 EOF。  
**当前证据：** 多次真实 run 日志中的 TLS EOF/Server disconnected；同一 key 的极简 tool call 返回 200。  
**是否需要修改 Harness：** 需要小范围兼容层，不需要重构。  
**推荐方案：** 保持 Chat Completions、vision override、较小 output token；加入请求大小/延迟/重试遥测和可恢复 checkpoint；在长实验前做真实 OpenHands payload smoke test。

### 瓶颈二：失败证据到 Capability 决策的 workflow

**为什么存在：** SDK 的 continuation 能防止文本立即结束，但不能替 Agent 识别“这是通用软件能力缺口”；当前任务可用已有 perception/control 工具完成，模型合理地可能选择继续改 Controller。  
**当前证据：** 7+ 次真实 physical trial 均无新 Capability；Task 2 曾尝试但越界 source 被安全拒绝；空能力库没有 capability 资产。  
**是否需要修改 Harness：** 不应再添加自研通用 loop；应优化 Agent-facing prompt、失败摘要结构和能力 schema 可发现性。  
**推荐方案：** 在每次失败 observation 中提供结构化“已尝试行为/未解决行为/是否像通用软件缺口”的事实字段；要求 Agent 在下一 physical trial 前作显式判断，但不替它选择 acquisition。

### 瓶颈三：新 Capability 的可验证真实闭环

**为什么存在：** 机制单测可证明 register/materialize，但最终目标要求真实 Agent 自主创建并实际使用；当前没有一条证据同时绑定 Agent decision、source digest、registration、materialization、Controller import、physical trial 和 persisted asset。  
**当前证据：** capability `072169...` 是预先/独立获得后被真实使用；Task 6 证明 use，不证明 autonomous creation。  
**是否需要修改 Harness：** 需要 provenance/ledger 展示增强和可恢复 acquisition 状态，但核心边界已经正确。  
**推荐方案：** 设计一个公开、通用且确实缺失的能力任务；预注册 acquisition acceptance schema；使用同一 durable conversation，验证每个链路节点和 physical provenance，禁止预置 Controller/手工 tool call。

## 6. 是否继续修改 Harness

建议：**冻结核心架构，停止继续扩展通用 Harness；只做两类小改动：Provider 兼容性/遥测和 acquisition provenance 可观测性。主要工作转向 Agent workflow 与实验设计。**

不建议重新设计架构，也不建议在 RoboForge 中实现第二套 Agent loop。当前 OpenHands/RoboForge/Adapter/evaluation 边界已经符合目标，继续大规模重构会增加风险并削弱可归因性。

## 7. 与 Direct Codex 的对比

Direct Codex 以前更容易完成任务，原因不是单纯模型更强，而是交互条件更好：

- **Interaction overhead：** Direct Codex 是短反馈、低延迟的人机循环；RoboForge 包含 RPC、Adapter reset、传感器处理、artifact 编码和长上下文，单步成本高。
- **Tool abstraction：** Codex 的 read/edit/run 是成熟原语；RoboForge 的 embodied 工具语义更复杂，Agent 要理解 physical budget、diagnostic 与 trial 差异。
- **Feedback visibility：** 软件 traceback 直接、紧凑；机器人反馈包含图像、深度、轨迹、验证器和几何关系，模型容易把局部改善误判为成功。
- **Code editing：** Direct Codex 常能连续修改并立即运行；RoboForge 受 Adapter Python 版本、Controller sandbox 和 physical trial 成本影响，迭代更慢。
- **Debugging：** Codex 看到明确 stack trace；RoboForge 需要从 evidence 推断抓取、支撑、可达性和控制时序，诊断难度更高。
- **Memory：** Direct Codex 依赖当前会话上下文；RoboForge 需要跨进程、跨任务、带 digest/provenance 的持久资产，检索与读取有额外步骤。
- **Experience reuse：** Codex 可直接复制代码；RoboForge 必须判断适用性、读取资产、物化并在真实环境中重新验证，正确但更慢。

## 8. 当前研究贡献判断

- **Infrastructure contribution：成立。** OpenHands-native embodied boundary、真实 Adapter/RPC、安全、不可变 evidence、authentic verification、恢复/idempotency 和跨 Python 运行时是可复现基础设施贡献。
- **Embodied coding agent contribution：部分成立。** 已有真实未知任务、观察、代码修改、physical trial、多模态证据和 OpenHands 工具循环；但尚未证明稳定自主完成未知任务的通用 Agent。
- **Self-improvement contribution：部分成立。** 真实 Controller digest 演化、trial comparison、Experience/Capability reuse 和 Task 8 authentic success 有证据；“自主发现缺口并获取新能力”仍缺失。

最薄弱部分是 self-improvement 的新能力获取闭环，其次是长 Provider 运行稳定性和跨任务成功率，而不是基础架构。

## 9. 未来 2-3 周路线

### 第 1 周：稳定性和可观测性

**目标：** 让长 OpenHands run 的失败可区分、可恢复。  
**修改内容：** 只增加请求大小/延迟/重试记录、conversation checkpoint 摘要、acquisition ledger 状态展示；不改变 Agent loop。  
**实验：** `gpt-5.6-sol` Chat Completions smoke、短多模态工具链、durable resume/idempotency matrix。  
**成功标准：** 失败能明确归因到 provider/SDK/Adapter；恢复不重放动作；所有状态可从 ledger 重建。

### 第 2 周：受控 Capability acquisition acceptance

**目标：** 构造一个确实需要通用软件能力、但不含任务坐标捷径的公开任务。  
**修改内容：** 完善 capability metadata/schema、source digest 和 provenance 展示；优化提示中的决策检查点。  
**实验：** 空能力库，同一 `gpt-5.6-sol` durable conversation，足够 Agent iterations 和有限 physical budget；禁止预置 capability、手工 tool call 和 evaluator 修改。  
**成功标准：** 一条证据链完整包含 need、Agent acquisition decision、new workspace `.py`、validation、registration、read、materialize、Controller import、physical use、persisted asset。

### 第 3 周：跨任务复用和论文级整理

**目标：** 验证新 capability/Experience 在第二个未知任务中的自主检索、适配和真实使用。  
**修改内容：** 只修复实验暴露的 provenance 或 workflow 可观测性问题；整理 acceptance matrix、失败分类和复现实验脚本。  
**实验：** 至少一个成功/一个失败的跨任务 pair，完整比较 Controller/evidence/asset lineage。  
**成功标准：** 成功任务有 authentic receipt；复用资产可追溯；失败不被包装为成功；报告可独立复现。

## 10. Go / No-Go

**Go，但采取“冻结核心、优化 workflow、补强实验验证”的 Go。**

下一步唯一最重要动作：**在 Provider 稳定窗口，用唯一验收模型 `gpt-5.6-sol` 运行一条专门设计为确实存在通用能力缺口的空能力库 durable acceptance trajectory，并逐项收集新 Capability 的完整 provenance 和 physical-use 证据。**

如果该实验仍未产生 acquisition，应先把结果分类为“任务不需要新能力”“Agent workflow 未触发”“Provider 截断”或“Capability 机制缺陷”，再针对分类行动；不能用增加预算或预置资产替代证据，也不应因此重写已验证的 OpenHands-native 基础设施。

## 附录：当前权威检查结果

- 全套回归：`279 passed, 2 skipped, 3 warnings in 91.19s`。
- OpenHands 原生测试：`13 passed`（SDK venv）。
- `compileall` 和 `git diff --check`：通过。
- 当前无运行中的 `roboforge`/`rpc_server`/LIBERO 进程。
- 当前工作树含未提交的新架构、文档和测试改动；本审计没有提交 commit，也没有覆盖或撤销这些改动。
- 最近真实 durable run 的物理 receipts 均为 false；历史 Task 8 `experiment://physical-000006` 仍是 authentic success 权威证据。
