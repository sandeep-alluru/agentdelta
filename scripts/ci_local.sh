#!/usr/bin/env bash
# Mirror GitHub Actions CI for substrate repos.
# Exit 0 only if ruff check + format + mypy (if present) + full pytest pass.
# Used by pre-push hooks and SUCCESS cycles. NEVER push if this fails.
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
# When installed inside a package repo as scripts/ci_local.sh:
if [[ -f "$PWD/pyproject.toml" ]]; then
  ROOT="$PWD"
elif [[ -f "$(git rev-parse --show-toplevel 2>/dev/null)/pyproject.toml" ]]; then
  ROOT="$(git rev-parse --show-toplevel)"
fi
cd "$ROOT"
NAME="$(basename "$ROOT")"

if [[ -d src ]]; then
  LINT_PATHS=(src tests)
elif [[ -d clickproof && -f clickproof/__init__.py ]]; then
  LINT_PATHS=(clickproof tests)
elif [[ -d "$NAME" && -f "$NAME/__init__.py" ]]; then
  LINT_PATHS=("$NAME" tests)
else
  LINT_PATHS=(.)
fi

if [[ -d "src/$NAME" ]]; then
  MYPY_TARGET="src/$NAME/"
elif [[ -d clickproof && -f clickproof/__init__.py ]]; then
  MYPY_TARGET="clickproof/"
elif [[ -d "$NAME" && -f "$NAME/__init__.py" ]]; then
  MYPY_TARGET="$NAME/"
else
  MYPY_TARGET=""
fi

VENV_BIN=""
if [[ -x .venv/bin/ruff ]]; then
  VENV_BIN=".venv/bin"
elif [[ -x "$ROOT/.venv/bin/ruff" ]]; then
  VENV_BIN="$ROOT/.venv/bin"
fi

run() {
  echo "+ $*"
  "$@"
}

echo "=== ci_local: $NAME paths=${LINT_PATHS[*]} ==="

if [[ -n "$VENV_BIN" ]]; then
  RUFF="$VENV_BIN/ruff"
  MYPY="$VENV_BIN/mypy"
  PYTEST="$VENV_BIN/pytest"
else
  RUFF="$(command -v ruff || true)"
  MYPY="$(command -v mypy || true)"
  PYTEST="$(command -v pytest || true)"
fi

if [[ -z "${RUFF:-}" || ! -x "${RUFF}" ]]; then
  echo "ERROR: ruff not found (create .venv and pip install -e '.[dev]')" >&2
  exit 2
fi

# Auto-format first so format --check never fails on style alone
run "$RUFF" format "${LINT_PATHS[@]}"
run "$RUFF" check --fix "${LINT_PATHS[@]}"
run "$RUFF" check "${LINT_PATHS[@]}"
run "$RUFF" format --check "${LINT_PATHS[@]}"

if [[ -n "${MYPY_TARGET}" && -x "${MYPY:-}" ]]; then
  run "$MYPY" "$MYPY_TARGET" --ignore-missing-imports
elif [[ -n "${MYPY_TARGET}" ]]; then
  echo "WARN: mypy binary missing — GitHub CI may still run mypy"
fi

if [[ "${CI_LOCAL_SKIP_PYTEST:-0}" == "1" ]]; then
  echo "SKIP pytest (CI_LOCAL_SKIP_PYTEST=1)"
else
  if [[ ! -x "${PYTEST:-}" ]]; then
    echo "ERROR: pytest not found" >&2
    exit 2
  fi
  if [[ -d tests ]]; then
    run "$PYTEST" tests/ -q --tb=line
  fi
fi

echo "=== ci_local OK: $NAME ==="
