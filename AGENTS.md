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

## Extend

1. Add fn to `providers/{convert,analyze}.py`
2. Export in `providers/__init__.py`
3. Add enum + match in `cli.py`
