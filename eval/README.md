# PDF Tool Evaluation Harness

Compare PDF extraction tools on the same documents.

## Quick Start

```bash
# Main supported CLI path: benchmark one PDF across providers
pdftoolkit benchmark docs/verkada.pdf

# Low-level eval harness
python -m eval.run_eval docs/verkada.pdf

# Run one optional eval tool directly
python -m eval.run_eval docs/verkada.pdf --tool mineru
```

`eval/` is the lower-level research harness. The supported user-facing command is `pdftoolkit benchmark`.

## Metrics

- **Extraction Quality**: Text completeness, structure preservation
- **Visual Detection**: Chart/table/figure identification accuracy
- **Speed**: Processing time per page
- **Resource Usage**: Memory, GPU utilization

## Tools Evaluated

| Tool | Category | Status |
|------|----------|--------|
| Docling | PDF→MD | Baseline |
| Marker | PDF→MD+Vision | Baseline |
| MinerU | PDF→MD | Eval |
| PDF-Extract-Kit | Layout/Tables | Eval |
| GOT-OCR2.0 | Unified OCR | Eval |
| Qwen2.5-VL | Document VLM | Eval |
| InternVL2.5 | Document VLM | Eval |
| PaddleOCR | Multilingual | Eval |
| olmOCR | Linearization | Eval |
| Nanonets OCR2 | Enterprise | Eval |
