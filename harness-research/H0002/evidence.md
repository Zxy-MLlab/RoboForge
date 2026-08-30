# Evidence

P30 produced 35 failed Tool results. Of those, 23/211 Tool round trips (10.9%)
were rejected solely because no open Decision Record existed:

- `run_command`: 6
- `write_file`: 6
- `run_controller`: 5
- `activate_shared_tool`: 2
- `replace_file_lines`: 2
- `reset_case`: 2

The exact post-compaction request schemas described what these tools did but
did not expose their consequence or Decision prerequisite. The compact system
prompt also did not state the prerequisite.

Generic baseline regression: a Tool registered with
`WORKSPACE_MUTATION` rendered only its supplied prose description. The model
contract contained no `WORKSPACE_MUTATION` or `record_decision` marker.
