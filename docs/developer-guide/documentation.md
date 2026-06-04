# Documentation

GeoBoost docs are built with MkDocs Material and deployed to GitHub Pages by
`.github/workflows/pages.yml`.

## Local Preview

```sh
uv sync --group docs --no-install-project
uv run --group docs --no-sync mkdocs serve
```

Build strictly before opening a PR:

```sh
uv run --group docs --no-sync mkdocs build --strict
```

## GitHub Pages

The Pages workflow:

1. Runs on pushes to `main`, pull requests, and manual dispatch.
2. Installs docs dependencies with `uv`.
3. Runs `mkdocs build --strict`.
4. Uploads the generated `site/` artifact on non-PR events.
5. Deploys with `actions/deploy-pages`.

In the repository settings, set Pages source to **GitHub Actions**.

## Quality Bar

Docs should be:

- Accurate to current code and tests.
- Explicit about alpha limits and native-only behavior.
- Example-driven, with copyable commands and code.
- Organized by task first, then reference details.
- Updated in the same PR as public API, CLI, artifact, splitter, schema, or
  validation changes.

## What To Update

| Change | Docs to check |
| --- | --- |
| Python estimator parameter or method | `user-guide/python-estimator.md`, `user-guide/parameters.md`, `reference/python-api.md`, `v1_api.md` |
| CLI command or config | `user-guide/cli.md`, `reference/cli.md`, `v1_api.md` |
| Model artifact | `model_artifact.md`, `developer-guide/extending.md` |
| Splitter or schema behavior | `user-guide/parameters.md`, `feature_schema.md`, `sparse_features.md` |
| Validation or release gate | `testing_strategy.md`, `developer-guide/build-test.md`, `v1_release_checklist.md` |

## Style

Prefer short task-oriented pages. Put broad claims behind reproducible evidence,
especially benchmark and model-quality claims.
