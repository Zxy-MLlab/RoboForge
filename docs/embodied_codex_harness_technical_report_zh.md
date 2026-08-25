# Embodied Codex Harness 技术说明

更新时间：2026-08-25
canonical kernel：`embodied_codex/`

## 1. 系统定义

Embodied Codex 是由底层 LLM 驱动的具身工程 Agent。LLM 是 Harness 的推理内核，
不是 Harness 外部的“另一个 GPT 脚本”。Harness 给它提供与 Coding Agent 类似的工程
权限和持久状态，同时用 Robot Adapter 隔离 evaluator 与仿真内部状态。

环境适配结束后，运行期间不需要人或外部 Codex 解决任务：

```text
LLM
 ├─ 读取任务、SDK、资产索引和历史证据
 ├─ 编写、测试、重写完整 controller.py
 ├─ 运行一次真实 episode
 ├─ 查看 RGB-D、动作回执、视频和 Tool 输出
 ├─ 维护 Capability Gap 与竞争假设
 ├─ 检索资产 / 搜索互联网 / 安装能力
 └─ 注册 Tool、Capability Package、Experience、Skill
```

Harness 不使用固定 Graph、Stage Node 或抓取模板。Controller 可以包含任意函数、
循环、分支、重新观测和恢复策略，只需通过稳定 Robot SDK 与环境交互。

## 2. 分层架构

### 2.1 Kernel Agent Loop

`kernel/agent_loop.py` 管理一次有界 session，而不替 LLM 决定机器人策略：

- 持久 workspace 与 iteration；
- 每轮最多一个真实 robot episode；
- episode 先事务提交，再允许模型后处理；
- 中断后从最后一份不可变证据恢复；
- 资产 Top-K 检索；
- 跨 case 编排由可选的通用 `CampaignRunner` 负责；
- Skill 保存通过显式资产工具完成；
- sealed evaluator 只由 `evaluation/` 外部 policy 调用。

主 LLM 直接获得完整任务语言和 SDK 合同，自主决定前置阶段，例如是否必须先开抽屉。

### 2.2 Coding Agent

`kernel/agent_loop.py` 把最小 system prompt、当前任务 JSON 和正式 function-call schema
发送给模型。
模型返回 tool calls，Harness 执行后把结构化结果送回模型。

工作记忆采用可重载压缩：图片在模型看过一次后保留路径/哈希而移除 base64；每次 Robot
RPC 都有稳定事件摘要，模型先读紧凑执行摘要，再用 `inspect_execution` 只展开
相关事件；大候选数组保留数量与有界 head，完整内容仍在不可变 artifact 中。每次请求记录：

- system prompt SHA256；
- messages SHA256；
- Tool schema SHA256；
- message 数量与字符数；
- 图像 payload SHA256；
- model 与 reasoning effort。

这样既避免上下文随 rollout 无限膨胀，也保留请求级可审计性。

### 2.3 Engineering Surface

`kernel/agent_loop.py` 注册的通用 function tools 是 LLM 的工程操作面：

- workspace：list/read/write/replace/run command；
- evidence：读取分页日志、提取视频关键帧、查看传感器图片；
- web：搜索公开网页/仓库，读取公开 HTTP(S) 页面；
- assets：检索、检查、注册、测试和修订资产；
- robot：静态 preflight 后运行一个完整 Controller。

Tool 默认以 dedicated manual + JSON Schema 为调用依据。确定性 Tool 注册只要求实现、
schema 和显式声明的公开来源 URL；Harness 自动生成 schema 一致的
基础 manual、依赖记录和来源 provenance。需要额外依赖的模型则必须走 Capability Package
并保留完整模型 provenance。`load_tool_source` 是独立、
分页的异常路径；manual 与证据冲突时才能用于定位实现，并通过证据发布新 manual revision。
资产检索由服务端返回有界摘要；模型不能用大 limit 枚举资产库。Gap 由 `record_gap` 保存
结构化失败记录。`latest_evidence`、
`executed_controller`、传感器文件和 hash-validated Tool manifest 都能作为正式证据引用。

### 2.4 Controller Runtime 与 Robot Adapter

`controller.py` 必须定义：

```python
def run(robot):
    ...
```

它在断网 bubblewrap 子进程中执行，只能调用：

```text
robot.observe(channel, request)
robot.use(tool_id, payload)
robot.act(action)
robot.verify(verifier, payload)
robot.record(event)
```

每个 Adapter 必须实现显式 `project_rpc_output()`。返回值按 RPC 正向 allowlist 投影，
未知字段 fail closed；不是依靠不断扩大的敏感字段黑名单。动作名、必填字段、范围和
opaque reference 均由机器可读 SDK 合同验证。

Adapter 不拥有任务策略。它只负责：

- 传感器与本体；
- 有界机器人动作；
- deployment-owned seed Tool；
- sensor-only verifier；
- controller I/O 封闭后的 evaluator barrier。

## 3. 能力资产系统

### 3.1 Tool

适合轻量确定性算法。源码和 vendored dependencies 被复制到 immutable version；
依赖必须精确 pin 并带 wheel SHA256。Tool 在独立断网进程运行，且只能读取 payload
显式引用的 sensor 文件。

### 3.2 Capability Package

用于不能压缩成单文件算法的能力：

- GPU perception model；
- grasp/robot policy；
- motion planner；
- 自包含、逐调用的 service wrapper；
- 带 checkpoint 的公开仓库。

需要 ROS graph、宿主 socket 或真机安全监督的能力不在隔离 worker 中冒充可执行资产；
它必须由环境 Adapter 以 deployment-owned Tool 绑定，并接受同一 Schema 和调用审计。
Harness 可以生成/检索该插件，但部署授权与 IPC 边界属于环境适配层。

Package 具有统一 `run(payload)` JSON contract，声明 `kind`、entrypoint、CPU/CUDA、
超时、依赖和 network policy。部署 worker 强制断网。Package 必须满足：

- 如果声明 checkpoint，则本地 bundle checkpoint 与声明 SHA256 一致；
- 所有依赖使用可验证的、固定版本的本地 wheel artifact。

公开来源、训练数据声明、model card、benchmark contamination 和 provenance 审计不属于
通用资产准入；需要时由 `evaluation/` 或部署方的独立 admission policy 执行。

带外部 import 的 package 必须使用带 hash 的 vendored lock，或声明当前 worker 中的
精确 `name==version` runtime requirement；运行时再次核对版本。

### 3.3 Experience

Experience 是带适用边界的条件性经验，不是自由日志。它保存 summary、applicability、
keywords，以及复制进资产目录的证据文件与哈希。跨任务仅通过 retrieval 注入 Top-K。

### 3.4 Capability Gap

Gap 是失败到能力获取的结构化研究记录：

```text
failure evidence
→ hypotheses
→ selected diagnosis
→ required capability contract
→ searched candidates
→ provenance decision
→ integration result
→ task validation
→ reuse evidence
```

revision 不可变且只能延伸最新 revision。状态生命周期受代码验证；`integrating` 必须
真的包含搜索候选、provenance 和集成记录，不能把普通参数微调冒充新能力。
同名 Experience retrieval 只返回最新 revision，已纠正的旧经验不会继续污染任务上下文。

### 3.5 Skill

Skill 由模型通过显式资产工具提交，核心只验证其成功证据和可移植性，并以
`candidate`/`verified`/`promoted` 状态保存。跨场景数量、generalization 或 sealed gate
由外部 admission/evaluation policy 决定。Skill 包含：

- 完整 Controller 与 SHA256；
- 实际调用的 Tool/Package 整包与 hash；
- Tool manual；
- 已复制且逐文件哈希的发展 execution、Adapter trace、rollout 与 Experience；
- preconditions/effects；
- required sensors/Robot operations；
- parameters/failure modes/composition notes。

LLM 可以提出接口，但 Harness 会把 Tool dependencies、Robot operations 和 sensors 与
真实成功 trace 合并，防止遗漏。新任务通过检索 Skill manifest 后按需分页读取 Controller
进行组合，不全量注入所有 Skill 源码。

## 4. 自主进化的一轮

1. Agent Loop 检索与任务/上轮失败最相关的 Tool、Skill、Experience、Gap。
2. GPT 读取持久 Controller 或创建新 Controller。
3. GPT 在沙箱中做语法、单元和合同测试。
4. preflight 拒绝可静态确定的 SDK 拼写、空 reference 等错误，不消耗 robot rollout。
5. Controller 在新 episode 执行；完整源代码先保存为 immutable snapshot。
6. `robot_execution.json`、视频、RGB-D、动作回执先原子提交。
7. GPT 查看证据，区分接口错误、感知错误、规划错误和物理失败。
8. 若现有能力不足，发布 Gap，搜索互联网，安装并测试新 Tool/Package。
9. GPT 改写 Controller，下一 iteration 再执行。
10. 模型可以选择保存 Skill、Experience 或 Gap；是否进入 sealed batch 由外部评测 runner 决定。

失败 iteration 的结果不会因为 GPT API 超时而丢失；恢复时不会重复已经完成的物理实验。

## 5. 反作弊与密钥安全

禁止输入：reward、done、success predicate、BDDL、privileged object pose/identity、
MuJoCo internal state、evaluation episode ID。

开发 case 使用私有 opaque handle，模型只看到 case 数量和同一程序的覆盖度。Harness
不能根据 state ID 选择 Controller，也禁止固定场景坐标。

进程环境使用 allowlist：Agent workspace、Controller、Tool 都看不到 `APEX_API_KEY`、
OpenAI/HuggingFace token 等宿主秘密，三者均在独立 network namespace 中断网。搜索、
网页读取和大文件下载由受控 broker 完成；下载只接受公网 HTTP(S)，限制大小、返回内容
SHA256 并写入 research ledger，随后才在 workspace 内离线解包、构建和注册。

sensor verifier 是开发信号，不替代最终 evaluator。sealed evaluation 先运行整个
Controller batch，再统一打开 evaluator；每个 episode 仅调用一次，结果不返回进化循环。

## 6. 运行与恢复

唯一端到端入口：

```bash
EMBODIED_CODEX_GROUNDINGDINO_CHECKPOINT=/path/to/groundingdino_swint_ogc.pth \
python evaluation/run_embodied_codex_libero.py \
  --tasks 0-9 --development-state 0 --development-state-count 3 \
  --unseen-state-count 3 \
  --max-iterations 16 --output runs/embodied_codex/libero_campaign
```

campaign 创建前就确定互不重叠的 development/sealed states。同一 Controller hash 必须
通过全部 development opaque cases 才能冻结。相同参数可恢复中断的 development；若 sealed batch
只完成一部分，则 fail closed，不把部分 evaluator 结果反馈给 Harness，也不在原目录续跑。

退出码：

- `0`：所有声明任务及 sealed states 成功；
- `1`：基础设施/进程失败；
- `2`：实验有效但能力未完成或 sealed evaluator 未全过。

## 7. 当前证据和边界

截至 2026-08-25，本机可复现的证据是：

- `python -m pytest -q`：97 passed；
- `python -m pytest -q evaluation`：12 passed；
- `compileall` 和 `git diff --check`：通过；
- `roboforge doctor --adapter libero`：通过，LIBERO、robosuite、感知依赖、checkpoint、
  Controller Runtime、Tool Runtime 和 sandbox 均完成真实 smoke test；
- 使用 `FakeModel` 的真实 LIBERO Adapter episode 已执行，但因模型生成了不受 SDK 支持的
  `set_value` 动作而失败。Harness 返回 `finished=false`、`completion_valid=false`、
  `resumable=true`，因此这不是 LIBERO 任务成功证据。

当前没有在本机完成真实 OpenAI/Apex 模型的 LIBERO 任务成功闭环，也没有完成完整 sealed
campaign；两者都必须在相应模型凭据和评测资源可用后单独验证，不能用 FakeModel 或 doctor
成功替代。
