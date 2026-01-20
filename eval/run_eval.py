#!/usr/bin/env python3
"""
Evaluation harness for comparing PDF extraction tools.

Usage:
    python -m eval.run_eval docs/sample.pdf
    python -m eval.run_eval docs/sample.pdf --tool mineru
    python -m eval.run_eval docs/sample.pdf --all
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


@dataclass
class EvalResult:
    """Result of running a single tool on a document."""

    tool: str
    input_file: str
    output_file: str | None
    success: bool
    error: str | None
    time_seconds: float
    output_size_bytes: int
    metadata: dict


def run_tool(
    name: str, func: Callable, input_path: Path, output_dir: Path
) -> EvalResult:
    """Run a tool and capture metrics."""
    start = time.time()
    output_file = None
    error = None
    success = False
    output_size = 0

    try:
        output_file = func(input_path, output_dir)
        if output_file and Path(output_file).exists():
            output_size = Path(output_file).stat().st_size
            success = True
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start

    return EvalResult(
        tool=name,
        input_file=str(input_path),
        output_file=str(output_file) if output_file else None,
        success=success,
        error=error,
        time_seconds=round(elapsed, 2),
        output_size_bytes=output_size,
        metadata={},
    )


# Tool implementations
def eval_docling(input_path: Path, output_dir: Path) -> str:
    """Evaluate Docling."""
    from docling.document_converter import DocumentConverter

    output_file = output_dir / f"{input_path.stem}-docling.md"
    converter = DocumentConverter()
    result = converter.convert(str(input_path))
    output_file.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return str(output_file)


def eval_marker(input_path: Path, output_dir: Path) -> str:
    """Evaluate Marker (without AI descriptions for speed)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    output_file = output_dir / f"{input_path.stem}-marker.md"
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(input_path))
    text, _, _ = text_from_rendered(rendered)
    output_file.write_text(text, encoding="utf-8")
    return str(output_file)


def eval_mineru(input_path: Path, output_dir: Path) -> str:
    """Evaluate MinerU."""
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

        output_file = output_dir / f"{input_path.stem}-mineru.md"

        with open(input_path, "rb") as f:
            pdf_bytes = f.read()

        local_image_dir = str(output_dir / "mineru_images")
        image_writer = DiskReaderWriter(local_image_dir)
        pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(local_image_dir, drop_mode="none")

        output_file.write_text(md_content, encoding="utf-8")
        return str(output_file)
    except ImportError:
        raise ImportError("MinerU not installed. Run: uv pip install magic-pdf")


def eval_got_ocr(input_path: Path, output_dir: Path) -> str:
    """Evaluate GOT-OCR2.0."""
    try:
        from transformers import AutoModel, AutoTokenizer

        output_file = output_dir / f"{input_path.stem}-got-ocr.md"

        tokenizer = AutoTokenizer.from_pretrained(
            "stepfun-ai/GOT-OCR2_0", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "stepfun-ai/GOT-OCR2_0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()

        # Convert PDF pages to images first
        from pdf2image import convert_from_path

        images = convert_from_path(str(input_path))

        results = []
        for i, img in enumerate(images):
            img_path = output_dir / f"_page_{i}.png"
            img.save(img_path)
            result = model.chat(tokenizer, str(img_path), ocr_type="format")
            results.append(f"## Page {i+1}\n\n{result}")
            img_path.unlink()  # Cleanup

        output_file.write_text("\n\n".join(results), encoding="utf-8")
        return str(output_file)
    except ImportError:
        raise ImportError(
            "GOT-OCR2.0 not installed. Run: uv pip install transformers pdf2image"
        )


def eval_paddleocr(input_path: Path, output_dir: Path) -> str:
    """Evaluate PaddleOCR PP-StructureV3."""
    try:
        from paddleocr import PPStructure

        output_file = output_dir / f"{input_path.stem}-paddleocr.md"

        engine = PPStructure(recovery=True, lang="en")
        result = engine(str(input_path))

        # Convert structure to markdown
        md_lines = []
        for page_result in result:
            for item in page_result:
                if item["type"] == "text":
                    md_lines.append(item["res"]["text"])
                elif item["type"] == "table":
                    md_lines.append(item["res"]["html"])
                elif item["type"] == "figure":
                    md_lines.append(f"[Figure: {item.get('bbox', 'unknown')}]")

        output_file.write_text("\n\n".join(md_lines), encoding="utf-8")
        return str(output_file)
    except ImportError:
        raise ImportError(
            "PaddleOCR not installed. Run: uv pip install paddleocr paddlepaddle"
        )


# Registry of available tools
TOOLS = {
    "docling": eval_docling,
    "marker": eval_marker,
    "mineru": eval_mineru,
    "got-ocr": eval_got_ocr,
    "paddleocr": eval_paddleocr,
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate PDF extraction tools")
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument(
        "-o", "--output", default="output/eval", help="Output directory"
    )
    parser.add_argument(
        "-t", "--tool", choices=list(TOOLS.keys()), help="Specific tool to run"
    )
    parser.add_argument("--all", action="store_true", help="Run all tools")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline tools only (docling, marker)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tools to run
    if args.tool:
        tools_to_run = {args.tool: TOOLS[args.tool]}
    elif args.baseline:
        tools_to_run = {k: v for k, v in TOOLS.items() if k in ("docling", "marker")}
    elif args.all:
        tools_to_run = TOOLS
    else:
        tools_to_run = {k: v for k, v in TOOLS.items() if k in ("docling", "marker")}

    # Run evaluations
    results = []
    for name, func in tools_to_run.items():
        print(f"Running {name}...")
        result = run_tool(name, func, input_path, output_dir)
        results.append(result)

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {status} ({result.time_seconds}s, {result.output_size_bytes} bytes)")

    # Save results
    results_file = output_dir / f"{input_path.stem}-results.json"
    with open(results_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Tool':<15} {'Status':<10} {'Time':>8} {'Size':>12}")
    print("-" * 50)
    for r in results:
        status = "OK" if r.success else "FAIL"
        size = f"{r.output_size_bytes:,}" if r.success else "-"
        print(f"{r.tool:<15} {status:<10} {r.time_seconds:>7.1f}s {size:>12}")

    return 0


if __name__ == "__main__":
    exit(main())
