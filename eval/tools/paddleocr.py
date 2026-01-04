"""PaddleOCR PP-StructureV3 evaluation script."""

from pathlib import Path


def eval_paddleocr(input_path: Path, output_dir: Path) -> str:
    """
    Evaluate PaddleOCR PP-StructureV3.

    Install: pip install paddleocr[doc-parser]

    Features:
    - 7-module pipeline (layout, table, formula, reading order, chart, markdown, preprocessing)
    - 20+ layout element categories
    - Multi-language support
    - CPU/GPU/NPU flexible deployment
    """
    from paddleocr import PPStructureV3

    output_file = output_dir / f"{input_path.stem}-paddleocr.md"

    pipeline = PPStructureV3()
    output = pipeline.predict(str(input_path))

    # Collect markdown from results
    md_parts = []
    for res in output:
        if hasattr(res, 'save_to_markdown'):
            temp_dir = output_dir / "_paddle_temp"
            temp_dir.mkdir(exist_ok=True)
            res.save_to_markdown(save_path=str(temp_dir))
            # Read generated markdown
            for md_file in temp_dir.glob("*.md"):
                md_parts.append(md_file.read_text())

    output_file.write_text("\n\n".join(md_parts), encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.paddleocr <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_paddleocr(input_path, output_dir)
    print(f"Output: {result}")
