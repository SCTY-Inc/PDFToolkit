# PDF Extraction Tools Comparison

Comprehensive evaluation of PDF extraction and document understanding tools.

## Quick Comparison

| Tool | Category | Best For | Install Size | GPU Required |
|------|----------|----------|--------------|--------------|
| **Docling** | PDF→MD | Simple extraction | ~500MB | No |
| **Marker** | PDF→MD+Vision | Layout preservation | ~2GB | Recommended |
| **MinerU** | PDF→MD | Complex documents | ~3GB | Recommended |
| **GOT-OCR2.0** | Unified OCR | Math, charts, music | ~1.2GB | Yes |
| **Qwen2.5-VL** | Document VLM | DocVQA, extraction | 7B: ~14GB | Yes |
| **InternVL2.5** | Document VLM | High accuracy | 8B: ~16GB | Yes |
| **PaddleOCR** | Full Pipeline | Multilingual | ~1GB | No |
| **olmOCR** | Academic | Papers, equations | ~2GB | Yes |
| **Nanonets OCR2** | Enterprise | Forms, signatures | ~6GB | Yes |

## Detailed Tool Profiles

### Currently Integrated

#### Docling (IBM)
```bash
pip install docling>=2.66.0
```
- **Strengths**: Simple API, fast, good baseline
- **Weaknesses**: Basic layout handling
- **Best for**: Quick text extraction

#### Marker-PDF
```bash
pip install marker-pdf>=1.10.0
```
- **Strengths**: Layout preservation, image extraction
- **Weaknesses**: Commercial license for >$2M revenue
- **Best for**: Documents with images needing AI descriptions

#### MegaParse (Quivr)
```bash
pip install megaparse>=0.0.55
```
- **Strengths**: Deep structural analysis
- **Weaknesses**: Heavy dependencies (unstructured)
- **Best for**: Complex document hierarchies

#### MarkItDown (Microsoft)
```bash
pip install markitdown[all]>=0.1.4
```
- **Strengths**: LLM integration, plugin architecture
- **Weaknesses**: Newer, less battle-tested
- **Best for**: LLM preprocessing pipelines

---

### Recommended Additions

#### MinerU (OpenDataLab)
```bash
pip install magic-pdf[full]
```
**DocVQA**: N/A | **Install**: ~3GB

```python
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

with open("doc.pdf", "rb") as f:
    pdf_bytes = f.read()

writer = DiskReaderWriter("./images")
pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, writer)
pipe.pipe_classify()
pipe.pipe_parse()
md = pipe.pipe_mk_markdown("./images", drop_mode="none")
```

- **Strengths**: Removes headers/footers, LaTeX formulas, 84+ languages
- **Weaknesses**: Complex setup, GPU recommended
- **Best for**: Academic papers, reports with equations

---

#### GOT-OCR2.0 (StepFun)
```bash
pip install transformers pdf2image
```
**Model**: `stepfun-ai/GOT-OCR-2.0-hf` (580M params)

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

inputs = processor(image, return_tensors="pt", format=True)
output = model.generate(**inputs, max_new_tokens=4096)
text = processor.decode(output[0], skip_special_tokens=True)
```

- **Strengths**: Unified model for text, math, tables, charts, music
- **Weaknesses**: Requires GPU
- **Best for**: Mixed content documents

---

#### Qwen2.5-VL (Alibaba)
```bash
pip install transformers torch accelerate
```
**Model**: `Qwen/Qwen2.5-VL-7B-Instruct` | **DocVQA**: 96.4%

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "doc.png"},
    {"type": "text", "text": "Extract all text in markdown format"}
]}]
inputs = processor.apply_chat_template(messages, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=2048)
```

- **Strengths**: Best DocVQA score, structured extraction, outperforms GPT-4o
- **Weaknesses**: Large model, slow inference
- **Best for**: High-accuracy document understanding

---

#### InternVL2.5 (OpenGVLab)
```bash
pip install transformers torch flash-attn
```
**Model**: `OpenGVLab/InternVL2_5-8B` | **DocVQA**: 95.1%

```python
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-8B", trust_remote_code=True)

outputs = model.generate(
    tokenizer.encode("<image>\nExtract text", return_tensors='pt'),
    images=[image], max_new_tokens=2048
)
```

- **Strengths**: Rivals GPT-4o, 4K image support, multiple sizes (1B-78B)
- **Weaknesses**: Trust remote code required
- **Best for**: Production document processing

---

#### PaddleOCR PP-StructureV3
```bash
pip install paddleocr[doc-parser]
```

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3()
output = pipeline.predict("doc.pdf")
for res in output:
    res.save_to_markdown(save_path="output")
```

- **Strengths**: 7-module pipeline, 20+ layout categories, CPU-friendly
- **Weaknesses**: Baidu ecosystem
- **Best for**: Multilingual documents, edge deployment

---

#### olmOCR (Allen AI)
```bash
pip install olmocr[gpu]
```

```bash
python -m olmocr.pipeline ./output --markdown --pdfs doc.pdf
```

- **Strengths**: Academic focus, reading order preservation, equation handling
- **Weaknesses**: GPU required, English-focused
- **Best for**: Scientific papers, technical documentation

---

#### Nanonets-OCR2
```bash
pip install transformers flash-attn
```
**Model**: `nanonets/Nanonets-OCR2-3B` | **DocVQA**: 85.15%

```python
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("nanonets/Nanonets-OCR2-3B")
```

- **Strengths**: Signatures, watermarks, checkboxes, Mermaid flowcharts
- **Weaknesses**: Smaller model = lower accuracy
- **Best for**: Enterprise forms, contracts

---

## Recommendation Matrix

| Use Case | Primary Tool | Fallback |
|----------|--------------|----------|
| Quick extraction | Docling | MarkItDown |
| Academic papers | MinerU | olmOCR |
| Forms/invoices | Nanonets OCR2 | Qwen2.5-VL |
| Charts/tables | Qwen2.5-VL | InternVL2.5 |
| Multilingual | PaddleOCR | MinerU |
| Math equations | GOT-OCR2.0 | MinerU |
| High accuracy | InternVL2.5-78B | Qwen2.5-VL-72B |
| Edge/mobile | PaddleOCR | Docling |

## Running Evaluations

```bash
# Baseline tools (already integrated)
python -m eval.run_eval docs/sample.pdf --baseline

# Specific tool
python -m eval.tools.mineru docs/sample.pdf
python -m eval.tools.qwen_vl docs/sample.pdf

# All tools (requires all dependencies)
python -m eval.run_eval docs/sample.pdf --all
```
