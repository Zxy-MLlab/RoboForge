# Evidence

In the exact P30 call-1 request, `def run(robot)` and `controller.py` each
occurred zero times. The `run_diagnostic` description only said it ran a
bounded read-only diagnostic Controller.

P30 diagnostic trajectory:

1. `run_diagnostic` before a Controller existed raised `FileNotFoundError`.
2. A later diagnostic execution used source with an invalid entrypoint and
   committed evidence with `ModuleNotFoundError`/incomplete execution.
3. Useful diagnostic evidence produced: zero.

The runtime error itself already stated the missing contract, proving the
information was safe to expose but available only after spending a model turn.
