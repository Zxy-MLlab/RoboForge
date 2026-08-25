# RoboForge Embodied Coding Agent Harness

RoboForge 是一个环境无关的 Embodied Coding Agent Harness。它提供持久 workspace、隔离
Controller runtime、Adapter SDK 合同、结构化证据、事务事件日志和渐进式 Tool/Skill/
Experience/Gap 检索；任务策略、失败诊断和能力获取由模型决定。

## 架构边界

`embodied_codex/kernel/` 是唯一运行底座：Agent Loop、workspace、sandbox runtime、
function-calling tools、capability manager、context builder、event store 和恢复检查点。
`embodied_codex/kernel/assets.py` 提供不可变 Tool
版本、Schema、manual-first 和资产库；`embodied_codex/deployments/` 是 Adapter 实现。
LIBERO 只通过 `deployments/libero.py` 和 `evaluation/run_embodied_codex_libero.py`
接入。反作弊、provenance、generalization 和 sealed evaluator 只属于 `evaluation/`
中的外部 Policy，不会被 kernel 导入。

旧实验代码已不再作为运行入口；正式 CLI、LIBERO 和自定义 Adapter 都进入同一个 Kernel。

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

LIBERO 是 optional Adapter。安装脚本固定 LIBERO、GroundingDINO、SAM 和 GraspNet
上游 revision；checkpoint 下载脚本在原子替换前校验 SHA256：

```bash
bash scripts/install_libero.sh
python scripts/download_libero_checkpoints.py
roboforge doctor --adapter libero
roboforge run --adapter libero --task 0 --profile autonomous \
  --run-dir runs/libero-task-0 --asset-root assets/shared
```

也可通过 `ROBOFORGE_GROUNDINGDINO_ROOT`、`ROBOFORGE_GROUNDINGDINO_CONFIG`、
`ROBOFORGE_GROUNDINGDINO_CHECKPOINT`、`ROBOFORGE_SAM_ROOT`、
`ROBOFORGE_SAM_CHECKPOINT`、`ROBOFORGE_GRASPNET_ROOT` 和
`ROBOFORGE_GRASPNET_CHECKPOINT` 使用外部安装目录。LIBERO 缺少依赖、源码、扩展、GPU
或 checkpoint 时，preflight 会列出每项失败并以非零状态退出，正式 run 不会继续。

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

可复现的容器入口使用同一个 `auto` backend。容器宿主机必须提供 Landlock，或允许镜像
内显式安装并探测通过 bubblewrap；Docker 本身不能替代 Controller 子进程的隔离探测：

```bash
docker build -t roboforge:0.5.0 .
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m \
  -v "$PWD/runs:/runs" -v "$PWD/assets:/assets" roboforge:0.5.0
```

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
