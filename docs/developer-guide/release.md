# Release Process

GeoBoost is currently alpha. Release readiness should be based on validation
evidence and explicit scope, not broad production claims.

## Pre-Release Checklist

1. Confirm the public scope in [Limitations](../limitations.md).
2. Run `just validate`.
3. Build docs with `uv run --group docs --no-sync mkdocs build --strict`.
4. Review model artifact compatibility.
5. Confirm Python, CLI, PyO3, and core docs match the code.
6. Explain any changed fixtures, golden files, model outputs, benchmark outputs,
   or generated images in the PR.

## Versioned Contracts

The main public contracts are:

- Python estimator constructor, methods, and error policy.
- CLI command names, options, output formats, and failure behavior.
- Native model artifact version.
- Weights artifact version.
- Feature schema payload shape.
- Sparse-set prediction requirements.

## Release Notes

Summarize changes by surface:

- Core Rust.
- CLI.
- PyO3 bindings.
- Python API.
- Docs.
- Tests.
- Fixtures.
- Benchmarks.
- Scripts.

Mention the validation commands that were run.
