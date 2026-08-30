# Change

Not implemented.

Candidate minimal direction, subject to a failing generic regression and
controlled comparison:

- return an authoritative exact-range digest from `read_file`;
- describe that value as the input for
  `replace_file_lines.expected_old_sha256`;
- preserve fail-closed mismatch behavior and optional backward compatibility.

No alternative editing API, automatic retry, or task policy is proposed.

Static audit further constrains the direction:

- the digest must use the exact same normalized range representation as the
  replacement check;
- it must describe a range guard, not whole-file identity;
- adding the result field must not change existing `content` or line-number
  semantics;
- CRLF normalization is existing behavior and is not part of this hypothesis.
