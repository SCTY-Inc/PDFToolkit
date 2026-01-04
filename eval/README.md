# PDF Tool Evaluation Harness

Compare PDF extraction tools on the same documents.

## Quick Start

```bash
# Run all evaluations
python -m eval.run_eval docs/verkada.pdf

# Run specific tool
python -m eval.run_eval docs/verkada.pdf --tool mineru

# Compare outputs
python -m eval.compare output/eval/
```

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
