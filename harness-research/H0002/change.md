# Change

`KernelTool.schema` now appends a mechanical suffix for consequences other
than `READ_ONLY` and `VALIDATION`:

`Consequence: <level>. Call record_decision before invoking this tool.`

The suffix is derived from the same authoritative metadata used by dispatch.
No parameter schema, handler, Decision lifecycle, safety check, or Tool
activation behavior changed.

In the generic core registry, 9/30 visible tools are marked. Added context is
731 characters (approximately 180 tokens) before provider tokenization.
