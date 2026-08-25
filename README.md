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

## 安全与稳定性

默认 `posix-hardened` sandbox 不需要 root、Docker、setuid binary 或 user namespace。
它实际启用 `no_new_privs`、libseccomp 网络/危险 syscall 阻断、rlimit、环境白名单、私有
临时目录和进程组超时回收；支持 Landlock 的内核会额外启用路径规则。工程命令只写独立
`staged_worktree/`，成功退出后由 Harness 检查文件类型、容量和冻结哈希并原子提交。
canonical workspace、checkpoint、events 和 evidence 不直接暴露为可写目标。

`--sandbox bubblewrap` 是探测成功后才可选择的 namespace 增强项，不是依赖。只有
`--profile dev --sandbox unsafe` 能显式使用无 syscall 隔离的调试模式；autonomous 和
benchmark 会拒绝 unsafe backend。

可复现的无特权容器入口使用同一个 POSIX backend，不安装或依赖 bubblewrap：

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
