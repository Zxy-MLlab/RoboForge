# Change

Not implemented.

Candidate minimal direction, subject to a failing generic regression and
controlled comparison:

- return an authoritative exact-range digest from `read_file`;
- describe that value as the input for
  `replace_file_lines.expected_old_sha256`;
- preserve fail-closed mismatch behavior and optional backward compatibility.

No alternative editing API, automatic retry, or task policy is proposed.
