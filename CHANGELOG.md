# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-17

### Added
- `AgentTrace` data model with JSONL save/load format
- `TraceNode` with content-addressed IDs (SHA-256) and embedding support
- `embed_trace()` using `all-MiniLM-L6-v2` for semantic node embeddings
- Sliding-window trace alignment via cosine similarity
- `diff_traces()` — finds first semantic fork point between two runs
- `ForkPoint` with human-readable divergence description
- `RedlineCallback` — LangChain/LangGraph-compatible callback handler
- `record()` context manager for one-line trace capture
- Rich terminal output (`print_diff`)
- JSON output (`to_json`) for programmatic consumption
- Markdown output (`to_markdown`) for GitHub PR comments
- CLI: `redline diff` and `redline inspect`
- GitHub Actions workflow for CI + behavioral diff PR comments
- 43 unit tests across all modules, 87% branch coverage

[Unreleased]: https://github.com/sandeep-alluru/redline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sandeep-alluru/redline/releases/tag/v0.1.0
