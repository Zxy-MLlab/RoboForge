# RoboForge-v2 验收证据索引

本文档索引当前可复核的闭环证据，不把确定性回归误称为真实机器人成功。

## 已验证的通用闭环

- `pytest -q`：268 passed（当前仓库回归）。
- `python -m compileall -q embodied_codex evaluation scripts`：通过。
- `git diff --check`：通过。
- `tests/live_acquisition_acceptance.py`：真实 HTTPS 搜索、下载、解包、构建、注册、契约测试、Controller 集成和 physical verification 全部执行。
- `tests/test_release_harness.py::test_generic_cli_end_to_end_recovery_multicase_and_cross_task_reuse`：独立 Task A/B workspace 使用共享 Skill/Experience/Tool，Task B 实际减少 physical executions。

## 真实 LIBERO 证据

历史 campaign 的完整报告位于：

`/root/autodl-tmp/experiments/libero-real-20260831-h0009/final-experiment-report.md`

其中记录了真实 LIBERO 多轮 Controller 演化、诊断、physical trial、证据持久化和 H0009 factual projection A/B。当前环境中的 doctor 需要使用项目的 Python 3.11 LIBERO 环境：

`source /root/autodl-tmp/roboforge_libero_env.sh && $ROBOFORGE_PYTHON -m embodied_codex doctor --adapter libero`

该检查已通过 adapter smoke、Controller runtime、sandbox、GPU、checkpoint 和 provider 检查。

最新 fresh autonomous campaigns 的事件与 checkpoint 位于：

`/root/autodl-tmp/experiments/goal-libero-task4/`

`/root/autodl-tmp/experiments/goal-libero-task7/`

Task 4 累计执行 15 次 physical trials，Task 7 累计执行 10 次。两条轨迹都包含 Controller 演化、artifact inspection、diagnostics、checkpoint recovery 和 immutable evidence；所有 authentic receipts 均为 `verified: false`。因此它们是完整失败/学习证据，不是成功证据。

## 仍未声称完成的项目

本仓库不把 FakeAdapter 的成功或历史 LIBERO 单任务成功，宣称为“当前新鲜运行已经证明真实 LIBERO 跨任务自主复用”。要满足该最强验收条件，还需要在可用的真实 LIBERO campaign 中保存一条 Task A 资产沉淀、Task B 自主检索并实际影响 Controller/experiment 的完整证据；任务成功率也必须由 authentic physical receipt 判定。
