# PDFToolkit

### PDF Extraction and Analysis CLI

<p>
<img alt="Python Version" src="https://img.shields.io/badge/python-3.11-blue.svg" />
<img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

-----

<p align="center">
 <a href="#-overview">Overview</a> •
 <a href="#-installation">Installation</a> •
 <a href="#-usage">Usage</a> •
 <a href="#-providers">Providers</a> •
 <a href="#-references">References</a>
</p>

-----

## Overview

PDFToolkit is a CLI for extracting and analyzing PDF content, with a focus on charts and visualizations. It provides a unified interface to multiple conversion and analysis backends.

## Installation

```bash
git clone https://github.com/amadad/pdftoolkit.git
cd pdftoolkit

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv && source .venv/bin/activate
uv sync
```

Set up API keys:
```bash
export OPENAI_API_KEY="..."      # For marker --describe, markitdown
export MISTRAL_API_KEY="..."     # For mistral provider
export TOGETHER_API_KEY="..."    # For together provider
```

Optional installs:
```bash
uv pip install megaparse unstructured[all-docs]==0.15.0  # megaparse provider
uv pip install together                                   # together provider
```

## Usage

### Convert PDF to Markdown

```bash
# Default provider (docling)
pdftoolkit convert document.pdf

# Choose provider
pdftoolkit convert document.pdf -p marker
pdftoolkit convert document.pdf -p mistral
pdftoolkit convert document.pdf -p markitdown
pdftoolkit convert document.pdf -p megaparse

# With options
pdftoolkit convert document.pdf -p marker --describe  # Add AI image descriptions
pdftoolkit convert document.pdf -o custom_output/     # Custom output directory
```

### Analyze Images/Charts

```bash
# Default provider (ollama - local)
pdftoolkit analyze chart.jpg

# Choose provider
pdftoolkit analyze chart.jpg -p ollama
pdftoolkit analyze chart.jpg -p together
pdftoolkit analyze chart.jpg -p colqwen

# With options
pdftoolkit analyze chart.jpg -q "What trends does this show?"
pdftoolkit analyze images/ --threshold 0.6  # Batch with confidence filter

# ColQwen returns relevance scores for queries
pdftoolkit analyze chart.jpg -p colqwen -q "chart showing growth"
```

### Help

```bash
pdftoolkit --help
pdftoolkit convert --help
pdftoolkit analyze --help
```

## Providers

### Convert Providers

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `docling` | IBM's document toolkit, basic extraction | Default |
| `marker` | PDF extraction with image support | `--describe` needs OPENAI_API_KEY |
| `mistral` | Mistral OCR API | MISTRAL_API_KEY |
| `markitdown` | Microsoft's converter | OPENAI_API_KEY |
| `megaparse` | Advanced structure parsing | Separate install |

### Analyze Providers

| Provider | Description | Requirements |
|----------|-------------|--------------|
| `ollama` | Local Llama Vision | Ollama running locally |
| `together` | Together API with confidence scoring | TOGETHER_API_KEY |
| `colqwen` | Visual similarity/relevance scores | Local GPU recommended |

## Project Structure

```
pdftoolkit/
├── cli.py              # Typer CLI
├── providers/
│   ├── convert.py      # PDF conversion providers
│   └── analyze.py      # Image analysis providers
├── clients.py          # API client singletons
└── utils.py            # Shared utilities
src/                    # Standalone scripts (reference implementations)
tests/                  # Test suite
```

## References

- [ColQwen2](https://huggingface.co/vidore/colqwen2-v0.1) - Visual retrieval model
- [Docling](https://github.com/DS4SD/docling) - IBM's document toolkit
- [Marker](https://github.com/VikParuchuri/marker) - PDF extraction
- [MegaParse](https://github.com/QuivrHQ/MegaParse) - Advanced parsing
- [MarkItDown](https://github.com/microsoft/markitdown) - Microsoft's converter
- [Mistral OCR](https://docs.mistral.ai/api/endpoint/ocr) - Mistral Document AI
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Together](https://together.ai/) - Cloud LLM inference
