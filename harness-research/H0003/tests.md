# Tests

Regression:

`test_controller_execution_tools_declare_source_entrypoint`

Baseline failed because neither description contained `controller.py` or
`def run(robot)`.

Post-change focused H0001-H0003 regressions: 3 passed. Related diagnostic and
model integration tests: 31 passed.

Primary experiment metric: completed diagnostic executions per attempted
diagnostic. Secondary metric: model calls before first valid execution.
