# H0005 candidate: Authoritative guarded line-edit digest

## Status

Candidate only. No production change, regression test, or embodied experiment
has been started.

## Problem

`replace_file_lines.expected_old_sha256` protects an edit by hashing the exact
old line range, including original line endings. The model-visible schema calls
it only a string, and `read_file` returns neither that digest nor byte-exact
range content. A model therefore cannot use the guard from authoritative public
data without independently guessing private normalization semantics.

## Hypothesis

Exposing the authoritative digest of the exact returned line range and
describing its relation to `expected_old_sha256` will reduce false
`WorkspaceError: file changed` rejections while preserving optimistic
concurrency protection.

## Stronger-model counterfactual

An ideal task strategist can omit the optional guard and edit successfully, so
this is not a hard execution block. However, it cannot infer whether the digest
covers the file, displayed content, or original range bytes, nor whether line
endings are included. The current interface makes the safer operation
unnecessarily difficult. Classification: H candidate, not A.

## Generality gates

- Task deletion: guarded text editing remains useful without any embodied task.
- Environment deletion: the persistent workspace is Adapter-independent.
- Model deletion: the ambiguity exists for every model/provider.
- Strategy independence: no Controller behavior or Tool preference is encoded.
- Real-robot realizability: this is workspace concurrency metadata only.
- Minimality: enrich the existing read result/schema; do not add an editing API.
