# CLAUDE.md

Project guidance for PDFToolkit.

## Setup

- Python: 3.11 (see `.python-version`)
- Package manager: `uv`

Common commands:

```bash
uv venv
source .venv/bin/activate
uv sync
uv sync --extra dev
```

## Tests

Run tests with:

```bash
uv run pytest
```

## Notes

- Use `trash` instead of `rm`
- Some tools (MegaParse, Together) are optional and installed separately
- OCR outputs are written to `output/`
