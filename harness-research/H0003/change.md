# Change

The `run_controller` and `run_diagnostic` descriptions now state that they
execute `workspace/controller.py`, which must define `def run(robot)`.

No file is created automatically. Runtime parsing, sandboxing, diagnostic
read-only enforcement, Decision requirements, physical budgets, and execution
semantics are unchanged.
