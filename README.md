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

PDFToolkit is a CLI for extracting, analyzing, and benchmarking PDF content, with a focus on charts and visualizations. It provides a unified interface to multiple conversion and analysis backends, plus a harness for comparing parsers on a single document.

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

### Benchmark a PDF Across Parsers

```bash
# Benchmark the integrated providers on one PDF
pdftoolkit benchmark document.pdf

# Benchmark a commercial-friendly subset
pdftoolkit benchmark document.pdf -t docling -t markitdown -t mistral

# Benchmark optional research tools (if installed)
pdftoolkit benchmark document.pdf -t mineru -t olmocr -t paddleocr
```

Outputs are written under `output/benchmark/<document-stem>/`, with a `results.json` summary and per-tool output directories.

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

### Benchmark Tools

`pdftoolkit benchmark` can run the integrated convert providers plus optional eval tools when installed.

| Tool | What it is | Commercial use |
|------|-------------|----------------|
| `docling` | IBM document parser | Yes |
| `markitdown` | Microsoft converter | Yes |
| `mistral` | Mistral OCR API | Yes |
| `megaparse` | Structural parser | Yes |
| `marker` | Layout-focused parser | Review license/weights |
| `paddleocr` | PP-Structure parser | Yes |
| `olmocr` | Technical-doc OCR | Yes |
| `mineru` | Strong open parser | No (AGPL) |
| `got-ocr`, `qwen-vl`, `internvl`, `nanonets` | VLM eval tools | Review model licenses |

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
├── benchmark.py        # Benchmark harness and tool registry
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
