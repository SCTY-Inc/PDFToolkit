# CLAUDE.md

Python 3.11 / uv

```bash
uv sync && uv run pytest
```

## CLI

```bash
pdftoolkit convert file.pdf [-p docling|marker|mistral|markitdown|megaparse] [-o dir] [--describe]
pdftoolkit analyze image.jpg [-p ollama|together|colqwen] [-q "query"] [--threshold 0.5]
```

## Structure

```
pdftoolkit/cli.py          # Typer entry
pdftoolkit/providers/      # convert.py, analyze.py
pdftoolkit/clients.py      # API singletons
src/                       # Reference scripts
```

## Notes

- `trash` not rm
- API keys: OPENAI, MISTRAL, TOGETHER
- Outputs: `output/`
