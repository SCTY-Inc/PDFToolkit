# Codemap

Generated: 2026-03-29

## Architecture

PDF extraction, image analysis, and single-document benchmarking CLI. `convert` turns PDFs into Markdown, `analyze` inspects images/charts, and `benchmark` runs one PDF across multiple providers with per-tool outputs and `results.json`.

## Directory Structure

| Path | Purpose | Key Files |
| --- | --- | --- |
| `pdftoolkit/` | Main package | `cli.py`, `benchmark.py`, `clients.py`, `utils.py` |
| `pdftoolkit/providers/` | Provider adapters | `convert.py`, `analyze.py` |
| `eval/` | Optional research/eval runners | `run_eval.py`, `tools/*.py` |
| `src/` | Reference scripts | `*.py` |
| `tests/` | Test suite | `test_cli.py`, `test_benchmark.py`, `test_*` |
| `docs/` | Sample PDFs | `*.pdf` |
| `output/` | Generated outputs | `benchmark/`, `*.md` |

## Entry Points

| Entry | File | Description |
| --- | --- | --- |
| CLI | `pdftoolkit/cli.py` | Typer app for `convert`, `benchmark`, and `analyze` |
| Benchmark registry | `pdftoolkit/benchmark.py` | Tool metadata, default selection, JSON result writing |

## Data Flow

- `pdftoolkit convert file.pdf` → selected convert provider → markdown output
- `pdftoolkit benchmark file.pdf` → tool registry → per-tool runs → `output/benchmark/<stem>/results.json`
- `pdftoolkit analyze image.jpg` → selected vision provider → text response or relevance scores

## Key Patterns

- **Provider selection**: enums in `cli.py` dispatch to functions in `pdftoolkit/providers/`
- **Benchmark registry**: `ToolSpec` records runner + commercial metadata + env requirements
- **Fail-fast validation**: CLI checks paths, env vars, and provider-specific flags before running
- **Client singletons**: shared API clients live in `pdftoolkit/clients.py`

## Dependencies (non-obvious)

| Package | Why |
| --- | --- |
| `docling` | default local PDF conversion |
| `markitdown` | commercial-friendly markdown conversion |
| `mistralai` | OCR API provider |
| `marker-pdf` | layout-focused conversion + optional figure descriptions |
| `ollama` | local vision analysis |
| `typer` / `rich` | CLI + terminal output |

## Common Tasks

| Task | Command |
| --- | --- |
| Convert one PDF | `pdftoolkit convert file.pdf -p docling` |
| Benchmark one PDF | `pdftoolkit benchmark file.pdf` |
| Benchmark explicit tools | `pdftoolkit benchmark file.pdf -t docling -t markitdown -t mistral` |
| Analyze a chart | `pdftoolkit analyze chart.jpg -p ollama` |
| Run tests | `uv run --extra dev pytest` |
| Add a provider | Add function in `providers/{convert,analyze}.py`, export in `providers/__init__.py`, update CLI enum or benchmark registry |
