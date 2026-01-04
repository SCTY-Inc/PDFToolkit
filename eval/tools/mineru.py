"""MinerU (magic-pdf) evaluation script."""

from pathlib import Path


def eval_mineru(input_path: Path, output_dir: Path) -> str:
    """
    Evaluate MinerU for PDF to Markdown conversion.

    Install: pip install magic-pdf[full]

    Features:
    - Removes headers/footers/page numbers
    - LaTeX formula conversion
    - Table to HTML
    - 84+ language OCR support
    """
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

    output_file = output_dir / f"{input_path.stem}-mineru.md"
    image_dir = output_dir / "mineru_images"
    image_dir.mkdir(exist_ok=True)

    with open(input_path, "rb") as f:
        pdf_bytes = f.read()

    image_writer = DiskReaderWriter(str(image_dir))
    jso_useful_key = {"_pdf_type": "", "model_list": []}

    pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
    pipe.pipe_classify()
    pipe.pipe_parse()
    md_content = pipe.pipe_mk_markdown(str(image_dir), drop_mode="none")

    output_file.write_text(md_content, encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.mineru <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_mineru(input_path, output_dir)
    print(f"Output: {result}")
