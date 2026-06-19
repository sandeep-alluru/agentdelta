# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `RegressionScore` dataclass and `compute_score()` — composite 0-100 behavioral similarity
  score with structural, semantic, tool-fidelity, and fork-penalty components for CI gates
- `to_html()` — self-contained HTML diff report with color-coded side-by-side step table,
  fork-point highlight, and embedded CSS (no external dependencies)
- `BatchDiffResult`, `batch_diff()`, and `batch_from_directory()` — diff multiple trace
  pairs in bulk; aggregate score and regression list across a directory of JSONL files
- CLI command `agentdelta score <baseline> <candidate>` — prints regression score and
  exits 0 (PASS/WARN) or 1 (FAIL) for easy CI integration
- Exports in `__init__.py`: `RegressionScore`, `compute_score`, `BatchDiffResult`,
  `batch_diff`, `batch_from_directory`, `to_html`

## [0.1.0] - 2026-06-17

### Added
- `AgentTrace` data model with JSONL save/load format
- `TraceNode` with content-addressed IDs (SHA-256) and embedding support
- `embed_trace()` using `all-MiniLM-L6-v2` for semantic node embeddings
- Sliding-window trace alignment via cosine similarity
- `diff_traces()` — finds first semantic fork point between two runs
- `ForkPoint` with human-readable divergence description
- `AgentdeltaCallback` — LangChain/LangGraph-compatible callback handler
- `record()` context manager for one-line trace capture
- Rich terminal output (`print_diff`)
- JSON output (`to_json`) for programmatic consumption
- Markdown output (`to_markdown`) for GitHub PR comments
- CLI: `agentdelta diff` and `agentdelta inspect`
- GitHub Actions workflow for CI + behavioral diff PR comments
- 43 unit tests across all modules, 87% branch coverage

[Unreleased]: https://github.com/sandeep-alluru/agentdelta/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sandeep-alluru/agentdelta/releases/tag/v0.1.0
