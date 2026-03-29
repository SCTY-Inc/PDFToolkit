# Agent Rules

- `trash` not rm; `uv` for Python
- Session end: gates pass → commit → `git push` (mandatory)

---

# Providers

## Convert (`pdftoolkit/providers/convert.py`)

| Provider | Key | Output |
|----------|-----|--------|
| docling (default) | - | `{stem}.md` |
| marker | OPENAI (--describe) | `{stem}/output.md` |
| mistral | MISTRAL | `{stem}-mistral.md` |
| markitdown | OPENAI | `{stem}-markitdown.md` |
| megaparse | - | `{stem}-megaparse.md` |

## Analyze (`pdftoolkit/providers/analyze.py`)

| Provider | Requirement |
|----------|-------------|
| ollama (default) | local Ollama |
| together | TOGETHER key |
| colqwen | local GPU |

## Benchmark (`pdftoolkit/benchmark.py`)

| Mode | Behavior |
|------|----------|
| default | Runs commercial-safe runnable tools: `docling`, plus `markitdown` with `OPENAI_API_KEY`, plus `mistral` with `MISTRAL_API_KEY` |
| explicit | Pass `-t/--tool` repeatedly to benchmark any registered provider |

## Extend

1. Add fn to `providers/{convert,analyze}.py`
2. Export in `providers/__init__.py`
3. Add enum + match in `cli.py`
