# Repository Guidelines

## Project Structure & Module Organization

Core Python package code lives in `opendm/`: model implementations are under
`model/`, data loading and transforms under `data/` and `dataset/`, training
logic under `trainer/` and `exp/`, and serving/inference code under `infer/`.
Runnable experiment configurations are in `playground/` (for example,
`playground/dm05_sft_demo.py`). Shell workflows belong in `script/`.
Integration clients and benchmark-specific code are isolated in
`third_party/`; documentation is in `docs/`, and sample images/data and robot
assets are in `assets/`. Keep generated checkpoints, normalization statistics,
and experiment logs out of source and commits.

## Build, Test, and Development Commands

Use Python 3.10+ and install the package in editable mode:

```bash
pip install -e .
pip install -e ".[fast-infer]"  # optional TensorRT backend
```

Run a training or evaluation entry point with its documented configuration,
typically through `torchrun`, for example:
`torchrun --nproc_per_node 1 playground/dm05_sft_demo.py --task train ...`.
Use the relevant guide in `docs/en/` for dataset, checkpoint, and GPU setup.
Start the inference service with the command in the inference guide, then
exercise its JSON API with `bash tests/curl_demo.sh` or
`bash tests/curl_history.sh`.

## Coding Style & Naming Conventions

Use four spaces, readable `snake_case` for functions and variables, `PascalCase`
for classes, and descriptive lowercase module names. Format and lint changed
Python files with `ruff format .` and `ruff check .`; the project configuration
uses an 88-character line length and double quotes. Keep dataset registrations
and robot-specific behavior explicit and localized rather than adding hidden
global state.

## Testing Guidelines

Pytest and pytest-cov are the declared test tools. Run `pytest` (or
`pytest --cov=opendm`) for unit tests when present. Name test files
`test_*.py` and test functions `test_*`. Because model and GPU integrations
may be unavailable in CI, also run the lightweight API curl scripts against a
local service when changing inference behavior.

## Commit & Pull Request Guidelines

Use concise imperative commits with the repository’s established prefixes:
`feat:`, `fix:`, `docs:`, or `chore:` (example: `fix: handle missing state`).
Pull requests should explain the motivation and affected workflow, link the
issue or benchmark when applicable, list validation commands and environment
requirements, and include API examples or screenshots for user-facing changes.
Call out checkpoint, dataset, hardware, or third-party dependency changes
explicitly.

## Security & Configuration Tips

Do not commit model weights, credentials, W&B tokens, private dataset paths, or
runtime logs. Review shell scripts and downloaded checkpoint destinations before
running them, and document any required environment variables in the relevant
guide.
