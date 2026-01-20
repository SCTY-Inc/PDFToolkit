import os
from pathlib import Path

from mistralai import Mistral


pdf_path = Path(
    os.getenv("PDF_PATH", "/Users/amadad/Desktop/care/2014-Consumption.pdf")
)
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{pdf_path.stem}-MistralOCR.md"

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    with pdf_path.open("rb") as pdf_file:
        uploaded = mistral.files.upload(
            file={
                "file_name": pdf_path.name,
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

print(f"Saved markdown to: {output_path}")
