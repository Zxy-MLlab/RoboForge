# RoboForge: OpenHands for the physical world

正式运行时是 `roboforge/`：OpenHands SDK 负责 Agent loop、LLM、conversation、
context、workspace、file editor、terminal、grep/glob、retry 与 session lifecycle；
RoboForge 只增加 physical experiment、Adapter boundary、immutable evidence、
authentic verification、physical budget、recovery/idempotency、embodied debugging
以及 Experience/Skill/Capability persistence。

```bash
python -m pip install -e '.[openhands]'
roboforge-openhands --adapter libero --adapter-python /path/to/libero-python \
  --task 0 --run-dir runs/libero-task-0 --asset-root assets/shared
```

OpenHands 1.44 使用 Python 3.12，而当前固定 LIBERO/MuJoCo deployment 使用
Python 3.11；正式入口通过带随机 token 的 mode-0600 Unix socket 启动隔离的
`roboforge-adapter-worker`。Adapter worker 是 sensor/action/reset/safety/verification
authority，OpenHands 进程不加载 LIBERO 或 evaluator。

`embodied_codex/kernel` 是旧版兼容和 embodied implementation 来源，不再是新正式
OpenHands-native CLI 的 Agent loop。新调用链不导入旧 AgentLoop、provider loop、
workspace、context builder 或 generic editor。

RoboForge 是一个环境无关的 Embodied Coding Agent Harness。它提供持久 workspace、隔离
Controller runtime、Adapter SDK 合同、结构化证据、事务事件日志和渐进式 Tool/Skill/
Experience/Gap 检索；任务策略、失败诊断和能力获取由模型决定。

## 架构边界

历史兼容入口中的 `embodied_codex/kernel/` 曾是运行底座：Agent Loop、workspace、sandbox runtime、
function-calling tools、capability manager、context builder、event store 和恢复检查点。
`embodied_codex/kernel/assets.py` 提供不可变 Tool
版本、Schema、manual-first 和资产库；`embodied_codex/deployments/` 是 Adapter 实现。
LIBERO 只通过 `deployments/libero.py` 和 `evaluation/run_embodied_codex_libero.py`
接入。反作弊、provenance、generalization 和 sealed evaluator 只属于 `evaluation/`
中的外部 Policy，不会被 kernel 导入。

旧实验代码已不再作为运行入口；正式 CLI、LIBERO 和自定义 Adapter 都进入同一个 Kernel。

完整的边界、证据模型、资产生命周期和迁移约束见
[`docs/embodied_codex_harness_technical_report_zh.md`](docs/embodied_codex_harness_technical_report_zh.md)。
其中 Generic Coding Harness 负责模型对话、上下文、workspace、文件和 shell；Kernel 的
embodied layer 负责 Physical Experiment、factual debugging evidence、within-task learning
ledger、Experience/Skill 持久化、Capability acquisition，以及 Adapter-owned observation、
action、reset、安全和物理验证。Adapter 不携带任务策略，资产也通过 progressive disclosure
按需检索和读取。

## 安装与运行

```bash
python -m pip install -e ".[test]"

roboforge doctor \
  --adapter embodied_codex.fake_adapter:FakeAdapter \
  --model embodied_codex.fake_adapter:FakeModel

roboforge run \
  --adapter embodied_codex.fake_adapter:FakeAdapter \
  --model embodied_codex.fake_adapter:FakeModel \
  --task "set target" --profile autonomous \
  --run-dir runs/fake-task --asset-root assets/shared
```

以上命令使用仓库内的确定性 Adapter/Model，但仍经过真实 CLI、Controller 子进程、Tool
Runtime、事件事务、checkpoint 和恢复链。自定义 Adapter 使用相同的
`package.module:AdapterClass` plugin 格式。

使用真实模型前必须显式配置对应 provider 的凭据：`OPENAI_API_KEY` 只连接
`https://api.openai.com/v1`，`APEX_API_KEY` 只连接 `https://api.apexin.ai/v1`。也可以在
doctor/run 中传入 `--provider openai` 或 `--provider apex`；内置 provider 不接受对方的
endpoint。

LIBERO 是 optional Adapter。安装脚本固定 LIBERO、GroundingDINO、SAM 和 GraspNet
上游 revision，并把源码安装到普通用户的数据目录；生成的
`~/.config/roboforge/libero_vendor.json` 会由 Adapter 自动读取。checkpoint 下载脚本在
原子替换前校验 SHA256。只安装 Python optional dependencies 可以执行：

```bash
python -m pip install -e ".[libero]"
```

完整安装和验证执行：

```bash
bash scripts/install_libero.sh
python scripts/download_libero_checkpoints.py
roboforge doctor --adapter libero
roboforge run --adapter libero --task 0 --profile autonomous \
  --run-dir runs/libero-task-0 --asset-root assets/shared
```

For the formal OpenHands-native CLI, select the model provider explicitly when
using a configured endpoint, for example `--provider apex --api-key-env
APEX_API_KEY`. The CLI forwards this provider configuration to the isolated
Adapter worker so LIBERO's authentic verifier uses the same approved endpoint.

完整 LIBERO deployment 使用 Python 3.10/3.11，并固定兼容的 robosuite 1.4.1 与
MuJoCo 2.3.7。Kernel 本身仍支持项目声明的其他 Python 版本。

上述 LIBERO run 需要先设置 `OPENAI_API_KEY` 或 `APEX_API_KEY`；没有模型凭据时 doctor
会明确报告 `model` 不可用并返回非零状态。

也可通过 `ROBOFORGE_GROUNDINGDINO_ROOT`、`ROBOFORGE_GROUNDINGDINO_CONFIG`、
`ROBOFORGE_GROUNDINGDINO_CHECKPOINT`、`ROBOFORGE_GROUNDINGDINO_TEXT_ENCODER`、
`ROBOFORGE_SAM_ROOT`、
`ROBOFORGE_SAM_CHECKPOINT`、`ROBOFORGE_GRASPNET_ROOT` 和
`ROBOFORGE_GRASPNET_CHECKPOINT` 使用外部安装目录。LIBERO 缺少依赖、源码、扩展、GPU
或 checkpoint 时，preflight 会列出每项失败并以非零状态退出，正式 run 不会继续。
上游 `LIBERO_CONFIG_PATH` 必须指向包含 `config.yaml` 的配置目录，而不是 YAML 文件本身。

Profile 只负责组合可选能力：`dev` 适合快速调试，`autonomous` 开启资产检索和沉淀，
`benchmark` 由外部评测 runner 额外加载研究 policy。自定义 Adapter 必须实现
`instruction`、`initial_observation`、`dispatch`、`project_rpc_output`、`sensor_report`、
`verification_receipt`、`execution_identity`、`resume_protocol`、
`validate_execution_receipt`、`register_capability` 和 `close`，详见
[`docs/adapter_authoring.md`](docs/adapter_authoring.md)。

Tool 的可选运行环境使用普通用户可创建的 Python `venv`。runtime spec 必须固定当前
Python implementation/version/ABI、平台、CPU（或宿主机兼容的 CUDA）以及每个依赖的
wheel 文件名、精确版本和 SHA256；wheel 先进入已校验的本地 CAS，再使用 `pip --no-index`
离线构建不可变 venv。未固定版本、缺少 wheel、SHA256 不匹配或需要 source build、root、
系统包、ROS、宿主机 socket、驱动或后台服务的 runtime 会 fail closed，并应改由
Adapter-owned deployment capability 提供。相同 runtime spec 的 Tool 共享同一个 runtime
ID，Tool 执行使用该 venv 的 Python，Harness Python 环境不会被安装过程修改。

## 安全与稳定性

### Embodied 操作

模型可使用稳定的高层操作：`observe` 获取当前公开环境状态，`run_controller` 执行一次
带完整 provenance 的 physical trial，`inspect_trial` 与 `compare_trials` 查看事实证据及
状态/行为差异；`find_capability`、`inspect_capability` 和 `acquire_capability` 支持渐进式
能力检索与获取。普通代码编辑和命令执行继续使用同一套 workspace/file/shell 工具。任务
策略、失败解释以及是否获取或复用能力始终由模型决定，Harness 负责实验账本、隔离、证据
和持久化。

真实 LIBERO campaign 的公开验收记录位于外部实验目录（例如
`/root/autodl-tmp/experiments/libero-real-20260831-h0009/final-experiment-report.md`）。
该报告区分了通用 FakeAdapter A/B、真实 LIBERO 多轮 trial，以及能力获取验收；其中
LIBERO 结果仅作为当前运行证据，不被硬编码进 Kernel，也不被宣称为普遍成功率提升。

默认 `auto` sandbox 先探测 rootless `posix-hardened`（Landlock + no_new_privs + seccomp +
rlimit + 环境白名单 + 私有临时目录 + 进程组回收），再探测 bubblewrap namespace 增强。
两者都必须通过真实的越权读写负向测试；没有文件系统隔离时 autonomous/benchmark 会
fail closed，绝不会降级为普通 subprocess。Linux 5.4 等没有 Landlock 且禁止 user
namespace 的系统必须明确报告不可安全运行。工程命令只写独立 `staged_worktree/`，成功
退出后由 Harness 检查文件类型、容量和冻结哈希并原子提交；canonical workspace、
checkpoint、events 和 evidence 不直接暴露为可写目标。

`--sandbox bubblewrap` 是探测成功后才可选择的 namespace 增强项，不是依赖。只有
`--profile dev --sandbox unsafe` 能显式使用无 syscall 隔离的调试模式；autonomous 和
benchmark 会拒绝 unsafe backend。

RPC 使用正向字段投影和严格 JSON 校验。Tool 版本以 SHA256 标识且不可变，调用前后执行
JSON Schema。Workspace 事务、事件日志和执行快照支持幂等提交与中断恢复。Adapter 负责
物理速度、力、工作空间和急停限制。

## 测试

```bash
python -m pytest -q tests evaluation
python -m compileall -q embodied_codex evaluation
git diff --check
python tests/live_acquisition_acceptance.py
```

确定性 kernel 测试不依赖 LIBERO、GPU 或真实 API，覆盖失败 Controller 读取证据、修改、
再次执行成功以及资产保存/复用。LIBERO benchmark 的正式报告和污染审计仍由外部 runner
负责，不能作为 Harness Core 的能力声明。
