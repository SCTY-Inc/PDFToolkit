from docling.document_converter import DocumentConverter
import os

source = "docs/ACCENTURE - How luxury brands are reinventing for success_CAIG.pdf"
output_dir = "output"
output_filename = os.path.splitext(os.path.basename(source))[0] + ".md"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

converter = DocumentConverter()
result = converter.convert(source)

# Save to markdown file
output_path = os.path.join(output_dir, output_filename)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())

print(f"Saved markdown to: {output_path}")
