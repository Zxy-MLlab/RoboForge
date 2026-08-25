# Harness 重构审计（2026-08-25）

## 归类

- Core：`kernel/` 的 Agent Loop、`kernel/workspace.py`、`kernel/runtime.py`、
  `kernel/context.py`、`kernel/events.py`、`kernel/recovery.py`；根目录的严格 RPC、
  sandbox、Schema 校验、Tool SHA256/版本和资产存储是可复用基础设施。
- Evaluation：`evaluation/run_embodied_codex_libero.py` 及反作弊、provenance、
  generalization、sealed evaluator policy。它们可以调用 Core，但 Core 不导入它们。
- 历史行为补丁：旧 `agent.py` 中 evidence/read/post-* 限额，旧 `engineering.py` 中
  repeated strategy、task fidelity、capability integration reviewer、强制 Gap acquisition，
  以及旧 `evolution.py` 中跨 case/controller hash gate。它们不在新 `kernel` 循环中执行。
- 必要工程稳定性：bubblewrap 隔离、RPC 正向投影和严格 JSON、Tool 内容哈希与不可变版本、
  workspace 路径限制、原子资产发布、fsync 事件日志、执行快照和恢复检查点。

## 迁移与删除

新 `kernel` 只执行模型显式请求的搜索、详情加载、workspace 变更、Controller 执行和资产
保存。旧复杂实现已移动到明确的 `embodied_codex/legacy/` 迁移层；canonical CLI、LIBERO
Adapter 和 benchmark runner 不依赖它。研究策略由 `evaluation/policy.py` 的可组合 hook
真实执行。渐进加载为 index → summary → manual/schema，源码必须通过显式 `load_source`
请求。旧 Graph/Stage Node/capability_library 和实验产物已从工作树移除。

## 依赖方向

```text
Evaluation / Benchmark Policy -> kernel + adapters
kernel -> interfaces, sandbox/runtime, assets
adapters -> kernel interfaces (deployment-owned implementation)
```

依赖方向测试扫描 `embodied_codex/kernel/*.py`，禁止出现 `evaluation` 或 `libero`。
