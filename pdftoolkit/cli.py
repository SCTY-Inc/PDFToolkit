"""CLI interface for PDFToolkit."""

import argparse
import sys
from pathlib import Path


def cmd_docling(args):
    """Convert PDF to markdown using Docling."""
    from docling.document_converter import DocumentConverter

    output_dir = Path(args.output or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(args.input).stem + ".md"
    output_path = output_dir / output_filename

    converter = DocumentConverter()
    result = converter.convert(args.input)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())

    print(f"Saved: {output_path}")


def cmd_marker(args):
    """Convert PDF to markdown with AI image descriptions using Marker."""
    import os
    from multiprocessing import freeze_support

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    from pdftoolkit.clients import get_openai_client, api_retry
    from pdftoolkit.utils import image_to_base64_raw

    freeze_support()

    output_dir = Path(args.output or f"output/{Path(args.input).stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    @api_retry
    def get_image_description(image_path):
        client = get_openai_client()
        encoded_image = image_to_base64_raw(image_path)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ],
            }],
            max_tokens=300,
        )
        return response.choices[0].message.content

    # Convert PDF
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"output_dir": str(output_dir)}
    )
    rendered = converter(args.input)
    text, metadata, images = text_from_rendered(rendered)

    # Process images
    output_dir_abs = output_dir.resolve()
    image_descriptions = {}

    for img_ref, img_data in images.items():
        safe_filename = os.path.basename(img_ref)
        if not safe_filename:
            continue

        img_path = output_dir / safe_filename
        if not img_path.resolve().is_relative_to(output_dir_abs):
            continue

        if isinstance(img_data, bytes):
            img_path.write_bytes(img_data)
        else:
            img_data.save(img_path, "JPEG")

        if not args.no_describe:
            description = get_image_description(img_path)
            image_descriptions[img_ref] = description
            print(f"Processed: {safe_filename}")

    # Enhance markdown
    enhanced_text = text
    for img_ref, description in image_descriptions.items():
        for marker in [f"![]({img_ref})", f"![{img_ref}]({img_ref})"]:
            if marker in enhanced_text:
                enhanced_text = enhanced_text.replace(
                    marker, f"{marker}\n\n*Image Description:* {description}\n"
                )
                break

    # Save
    output_file = output_dir / "enhanced_output.md"
    output_file.write_text(enhanced_text, encoding="utf-8")
    print(f"Saved: {output_file}")


def cmd_megaparse(args):
    """Parse PDF using MegaParse."""
    from megaparse import MegaParse
    from megaparse.parser.unstructured_parser import UnstructuredParser

    output_dir = Path(args.output or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(args.input).stem + "-megaparse.md"
    output_path = output_dir / output_filename

    parser = UnstructuredParser()
    megaparse = MegaParse(parser)
    megaparse.load(args.input)
    megaparse.save(str(output_path))

    print(f"Saved: {output_path}")


def cmd_markitdown(args):
    """Convert PDF using MarkItDown."""
    from markitdown import MarkItDown
    from openai import OpenAI

    output_dir = Path(args.output or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(args.input).stem + "-markitdown.md"
    output_path = output_dir / output_filename

    client = OpenAI()
    md = MarkItDown(llm_client=client, llm_model="gpt-4")
    result = md.convert(args.input)

    output_path.write_text(result.text_content, encoding="utf-8")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="pdftoolkit",
        description="PDF extraction and analysis toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Docling
    p_docling = subparsers.add_parser("docling", help="Basic PDF to markdown")
    p_docling.add_argument("input", help="Input PDF file")
    p_docling.add_argument("-o", "--output", help="Output directory")
    p_docling.set_defaults(func=cmd_docling)

    # Marker
    p_marker = subparsers.add_parser("marker", help="PDF to markdown with AI image descriptions")
    p_marker.add_argument("input", help="Input PDF file")
    p_marker.add_argument("-o", "--output", help="Output directory")
    p_marker.add_argument("--no-describe", action="store_true", help="Skip image descriptions")
    p_marker.set_defaults(func=cmd_marker)

    # MegaParse
    p_mega = subparsers.add_parser("megaparse", help="Advanced document structure parsing")
    p_mega.add_argument("input", help="Input PDF file")
    p_mega.add_argument("-o", "--output", help="Output directory")
    p_mega.set_defaults(func=cmd_megaparse)

    # MarkItDown
    p_mit = subparsers.add_parser("markitdown", help="Microsoft MarkItDown conversion")
    p_mit.add_argument("input", help="Input PDF file")
    p_mit.add_argument("-o", "--output", help="Output directory")
    p_mit.set_defaults(func=cmd_markitdown)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
