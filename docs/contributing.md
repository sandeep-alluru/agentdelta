# Contributing

Thank you for considering a contribution to redline!

For the full contribution guide — setup, workflow, PR conventions, review timeline — see [CONTRIBUTING.md](../CONTRIBUTING.md) in the repo root.

## Quick links

- [Bug report](https://github.com/sandeep-alluru/redline/issues/new?template=bug_report.yml)
- [Feature request](https://github.com/sandeep-alluru/redline/issues/new?template=feature_request.yml)
- [Architecture overview](architecture.md)
- [Adding a framework adapter](adding-adapter.md)

## Dev setup

```bash
git clone https://github.com/sandeep-alluru/redline
cd redline
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Make targets

```bash
make test       # pytest with branch coverage (fails under 85%)
make lint       # ruff check + format check
make typecheck  # mypy
make fmt        # ruff format (auto-fix)
make all        # lint + typecheck + test
```
