# Rust Crates Agent Guide

## Dev environment tips
- This folder contains the Rust workspace crates.
- Keep shared behavior in `cartoboost-core`, CLI-only behavior in `cartoboost-cli`, and PyO3 binding code in `cartoboost-py`.
- Use workspace dependencies and lint settings from the root `Cargo.toml`.

## Testing instructions
- Run `cargo fmt --all --check`.
- Run `cargo clippy --workspace --all-targets -- -D warnings`.
- Run `cargo test --workspace`.

## PR instructions
- Name the crate or crates affected.
- Mention any Python, CLI, or artifact compatibility impact from Rust changes.
