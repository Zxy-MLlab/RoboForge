# Change

Implemented in `2e6f423`:

- return an authoritative exact-range digest from `read_file`;
- describe that value as the input for
  `replace_file_lines.expected_old_sha256`;
- preserve fail-closed mismatch behavior and optional backward compatibility.

No alternative editing API, automatic retry, task policy, or stale-write
weakening was introduced.

Static audit further constrains the direction:

- the digest must use the exact same normalized range representation as the
  replacement check;
- it must describe a range guard, not whole-file identity;
- adding the result field must not change existing `content` or line-number
  semantics;
- CRLF normalization is existing behavior and is not part of this hypothesis.

The returned `range_sha256` is computed through the same normalized range
representation used by replacement. The Tool schema/manual states that
`expected_old_sha256` must be copied from the most recent same-range
`read_file`. The guard remains optional, range-scoped, and fail closed.

Rejected alternatives remain recorded in `alternatives.md`.
