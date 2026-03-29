# CLAUDE.md

Python 3.11 / uv

```bash
uv sync && uv run pytest
```

## CLI

```bash
pdftoolkit convert file.pdf [-p docling|marker|mistral|markitdown|megaparse] [-o dir] [--describe]
pdftoolkit benchmark file.pdf [-t tool ...] [-o dir]
pdftoolkit analyze image.jpg [-p ollama|together|colqwen] [-q "query"] [--threshold 0.5]
```

## Structure

```
pdftoolkit/cli.py          # Typer entry
pdftoolkit/benchmark.py    # Benchmark harness + registry
pdftoolkit/providers/      # convert.py, analyze.py
pdftoolkit/clients.py      # API singletons
src/                       # Reference scripts
```

## Notes

- `trash` not rm
- API keys: OPENAI, MISTRAL, TOGETHER
- `convert --describe` only works with `-p marker`
- `benchmark` defaults are env-aware: always `docling`, plus `markitdown` with `OPENAI_API_KEY`, plus `mistral` with `MISTRAL_API_KEY`
- Outputs: `output/`
