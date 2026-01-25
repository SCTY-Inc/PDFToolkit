"""PDF conversion providers."""

import os
from pathlib import Path

from pdftoolkit.clients import get_openai_client, api_retry
from pdftoolkit.utils import image_to_base64_raw, safe_output_path


def convert_docling(input_path: Path, output_dir: Path) -> Path:
    """Convert PDF to markdown using Docling."""
    from docling.document_converter import DocumentConverter

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.md"

    converter = DocumentConverter()
    result = converter.convert(str(input_path))

    output_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return output_path


def convert_marker(input_path: Path, output_dir: Path, describe: bool = False) -> Path:
    """Convert PDF to markdown with optional AI image descriptions using Marker."""
    from multiprocessing import freeze_support

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    freeze_support()

    output_dir = output_dir / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    @api_retry
    def get_image_description(image_path: Path) -> str:
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
    rendered = converter(str(input_path))
    text, metadata, images = text_from_rendered(rendered)

    # Process images
    image_descriptions = {}

    for img_ref, img_data in images.items():
        img_path = safe_output_path(output_dir, img_ref)
        if img_path is None:
            continue

        if isinstance(img_data, bytes):
            img_path.write_bytes(img_data)
        else:
            img_data.save(img_path, "JPEG")

        if describe:
            description = get_image_description(img_path)
            image_descriptions[img_ref] = description
            print(f"Processed: {img_path.name}")

    # Enhance markdown with descriptions
    enhanced_text = text
    for img_ref, description in image_descriptions.items():
        for marker in [f"![]({img_ref})", f"![{img_ref}]({img_ref})"]:
            if marker in enhanced_text:
                enhanced_text = enhanced_text.replace(
                    marker, f"{marker}\n\n*Image Description:* {description}\n"
                )
                break

    output_path = output_dir / "output.md"
    output_path.write_text(enhanced_text, encoding="utf-8")
    return output_path


def convert_megaparse(input_path: Path, output_dir: Path) -> Path:
    """Parse PDF using MegaParse."""
    from megaparse import MegaParse
    from megaparse.parser.unstructured_parser import UnstructuredParser

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-megaparse.md"

    parser = UnstructuredParser()
    megaparse = MegaParse(parser)
    megaparse.load(str(input_path))
    megaparse.save(str(output_path))

    return output_path


def convert_markitdown(input_path: Path, output_dir: Path) -> Path:
    """Convert PDF using MarkItDown."""
    from markitdown import MarkItDown

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-markitdown.md"

    md = MarkItDown(llm_client=get_openai_client(), llm_model="gpt-4")
    result = md.convert(str(input_path))

    output_path.write_text(result.text_content, encoding="utf-8")
    return output_path


def convert_mistral(input_path: Path, output_dir: Path) -> Path:
    """Convert PDF using Mistral OCR."""
    from mistralai import Mistral

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-mistral.md"

    with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
        with input_path.open("rb") as pdf_file:
            uploaded = mistral.files.upload(
                file={
                    "file_name": input_path.name,
                    "content": pdf_file,
                },
                purpose="ocr",
            )

        ocr_response = mistral.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "file",
                "file_id": uploaded.id,
            },
        )

        markdown = "\n\n".join(page.markdown for page in ocr_response.pages)
        output_path.write_text(markdown, encoding="utf-8")

    return output_path
