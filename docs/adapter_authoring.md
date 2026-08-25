# Robot Adapter 接入合同

Embodied Codex 的任务学习内核与 benchmark/真机解耦。新环境只需实现
`embodied_codex.interfaces.RobotDeployment`；适配器不得包含任务求解策略。

## 最小接口

```python
class MyRobotDeployment:
    instruction: str

    def initial_observation(self) -> dict:
        """返回任务启动时的传感器观察或 artifact reference。"""

    def dispatch(self, method: str, arguments: dict) -> dict:
        """实现 observe/use/act/verify/record。"""

    def project_rpc_output(self, method: str, arguments: dict,
                           result: dict) -> dict:
        """按 method 正向投影允许返回给 Controller 的字段。未知字段必须拒绝。"""

    def register_capability(self, tool_id: str, function, contract: dict) -> None:
        """原子绑定 Harness 新增 Tool 的隔离调用函数及 input/output Schema。"""

    def sensor_report(self, execution: dict) -> dict:
        """返回独立于 Controller 自述的传感器验证和诊断证据。"""

    def verification_receipt(self, execution: dict) -> dict:
        """绑定 verified、Controller SHA、episode 和 environment generation。"""

    def execution_identity(self) -> dict:
        """返回 episode_id 和 environment_generation。"""

    def resume_protocol(self) -> dict:
        """声明 resume token、重放和动作幂等能力；不支持恢复时明确返回 false。"""

    def validate_execution_receipt(self, receipt: dict) -> bool:
        """仅在当前物理状态仍与收据一致时返回 true。"""

    def close(self) -> None:
        """释放机器人/仿真资源并落盘 trace。"""
```

Adapter factory 每次调用必须创建一个新 episode。Harness 只保存 factory，不持有
benchmark 内部对象。

`project_rpc_output()` 不是字段黑名单：它必须按 method 正向选择公开字段。Kernel 只验证
RPC 是严格 JSON 和字段符合通用 RPC envelope，不知道 benchmark 字段。reward、done、
benchmark state 等泄漏检查必须由独立 Evaluation Policy 执行。

## SDK 合同

适配器需要向 Harness 提供机器可读合同：

```json
{
  "protocol": "my-robot-sdk-v1",
  "methods": {
    "observe": {}, "use": {}, "act": {}, "verify": {}, "record": {}
  },
  "actions": {
    "move_to_pose": {
      "required": ["type", "pose_ref"],
      "optional": {"offset": "xyz", "max_steps": "integer"}
    }
  },
  "verifiers": {
    "visual_goal_relation": {"required": ["frame"], "optional": {}}
  },
  "opaque_reference_fields": ["point_ref", "pose_ref"]
}
```

Controller 的静态 preflight 和运行时 Adapter 都必须执行这个合同。不要在 prompt 中
同时提供一套别名；一个动作只有一个 canonical enum。

## Opaque reference

感知/规划 Tool 可以输出世界坐标供诊断，但动作只能消费 Adapter 签发的 live opaque
reference。Adapter 保存 reference 到本 episode 传感器/Tool provenance 的私有映射：

- 伪造或空 reference：拒绝；
- 上一 episode 的 reference：拒绝；
- 重新观测导致证据失效时，Controller 应重新获取；
- reference 不得编码 benchmark object ID/state ID。

## Tool 绑定

Adapter 可注册环境部署已有的 seed Tool，例如相机感知服务或真机 MoveIt 服务。每个
Tool 必须提供 input/output JSON Schema、manual、内容哈希、依赖和测试状态。Agent 自主注册的
analytic Tool/Capability Package 由 Harness 以隔离 worker 执行，再通过同一个
`robot.use` RPC 暴露；Adapter 不需要为每个新 Python 算法写特殊分支。

当冻结 Skill 迁移到新环境时，deployment-owned Tool 按 capability name 绑定到当前
版本；Harness 只结构化替换完全相等的 Tool ID 字符串，不改 Controller 控制流。

## Evaluator barrier

以下规则只属于外部 Evaluation Policy，不是通用 Adapter 或 Kernel 合同。

开发阶段：

- `sensor_report` 只能使用允许传感器；
- reward/done/check_success 不可进入 Controller、LLM evidence 或 Tool payload；
- sensor verifier 误差必须与最终 evaluator 分开报告。

sealed 阶段：

1. 预先声明全部 case；
2. 冻结同一 Controller 和全部资产 hash；
3. 运行完整 batch，并封闭所有 Controller I/O；
4. 最后逐 episode 调一次 evaluator；
5. evaluator 结果写入 `_evaluator_only/`，不得反馈进化。

## 真机建议

- `act` 必须有速度、力、workspace、超时和急停限制；
- 物理动作前保留 Adapter 级 safety supervisor；
- Harness 的反作弊隔离不替代真机安全认证；
- 首次接入先用无动作/低速 conformance Adapter 验证 RPC、日志和关闭语义。

## 发布前 conformance

新 Adapter 至少证明：

- 完整 Controller 由模型生成并执行；
- RPC 输出全部经过 allowlist；
- Controller/Tool 看不到宿主密钥和 evaluator；
- 失败 episode 能事务恢复；
- 同一 Controller hash 能运行多个 opaque case；
- Tool Schema 在调用前后验证；
- trace、视频、传感器和 Controller snapshot 均可审计；
- evaluator 只在 sealed batch barrier 后打开。
