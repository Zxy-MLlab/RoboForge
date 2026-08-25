# RoboForge Embodied Coding Agent Harness

RoboForge 是一个环境无关的 Embodied Coding Agent Harness。它提供持久 workspace、隔离
Controller runtime、Adapter SDK 合同、结构化证据、事务事件日志和渐进式 Tool/Skill/
Experience/Gap 检索；任务策略、失败诊断和能力获取由模型决定。

## 架构边界

`embodied_codex/kernel/` 是唯一运行底座：Agent Loop、workspace、sandbox runtime、
function-calling tools、capability manager、context builder、event store 和恢复检查点。
`embodied_codex/assets.py` 提供不可变 Tool
版本、Schema、manual-first 和资产库；`embodied_codex/deployments/` 是 Adapter 实现。
LIBERO 只通过 `deployments/libero.py` 和 `evaluation/run_embodied_codex_libero.py`
接入。反作弊、provenance、generalization 和 sealed evaluator 只属于 `evaluation/`
中的外部 Policy，不会被 kernel 导入。

旧实验代码已不再作为运行入口；正式 CLI、LIBERO 和自定义 Adapter 都进入同一个 Kernel。

## 安装与运行

```bash
cd RoboForge
python -m pip install -e '.[test]'
# LIBERO adapter and robot simulation dependencies
python -m pip install -e '.[libero]'

# 快速调试自定义 Adapter
embodied_codex run --adapter my_package:MyAdapter \
  --task 'put the bowl on the plate' --profile dev \
  --run-dir runs/task_b --asset-root assets/shared \
  --model my_package:FakeModel

# 自主模式使用 API 模型
export OPENAI_API_KEY=...
embodied_codex run --adapter my_package:MyAdapter --task 'put the bowl on the plate' \
  --profile autonomous

# LIBERO 兼容 Adapter
roboforge run --adapter libero --task 0 --profile autonomous \
  --run-dir runs/libero_task0 --asset-root assets/shared

# 运行环境检查
roboforge doctor --adapter libero
```

LIBERO perception checkpoints and third-party sources are intentionally external. Set
`ROBOFORGE_GROUNDINGDINO_ROOT`, `ROBOFORGE_GROUNDINGDINO_CONFIG`,
`ROBOFORGE_GROUNDINGDINO_CHECKPOINT`, `ROBOFORGE_SAM_ROOT`, `ROBOFORGE_SAM_CHECKPOINT` and
`ROBOFORGE_GRASPNET_CHECKPOINT`; `roboforge doctor --adapter libero` performs adapter and
proprioception smoke checks before a run.

Profile 只负责组合可选能力：`dev` 适合快速调试，`autonomous` 开启资产检索和沉淀，
`benchmark` 由外部评测 runner 额外加载研究 policy。自定义 Adapter 至少实现
`instruction`、`dispatch`、`project_rpc_output`、`sensor_report` 和 `close`，详见
[`docs/adapter_authoring.md`](docs/adapter_authoring.md)。

## 安全与稳定性

Controller、工程命令和获取的 Tool 在断网 bubblewrap worker 中运行；RPC 使用正向字段
投影和严格 JSON 校验。Tool 版本以 SHA256 标识且不可变，调用前后执行 JSON Schema。
Workspace 事务、事件日志和执行快照支持幂等提交与中断恢复。Adapter 负责物理速度、力、
工作空间和急停限制。

## 测试

```bash
python -m pytest -q tests evaluation
python -m compileall -q embodied_codex evaluation
git diff --check
```

确定性 kernel 测试不依赖 LIBERO、GPU 或真实 API，覆盖失败 Controller 读取证据、修改、
再次执行成功以及资产保存/复用。LIBERO benchmark 的正式报告和污染审计仍由外部 runner
负责，不能作为 Harness Core 的能力声明。
