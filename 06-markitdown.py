from markitdown import MarkItDown
from openai import OpenAI
from pathlib import Path

# Initialize OpenAI client and MarkItDown
client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4")


def convert_pdf_with_descriptions(pdf_path):
    """Convert PDF to markdown with inline image descriptions."""
    print(f"\nProcessing {pdf_path}...")

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Convert PDF and get result object
        result = md.convert(pdf_path)

        # Generate output filename
        output_file = output_dir / f"{Path(pdf_path).stem}-Markitdown.md"

        # Write content to file
        with open(output_file, "w") as f:
            f.write(result.text_content)

        print(f"\nSuccessfully converted {pdf_path} to {output_file}")

    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")


# Process PDF
pdf_file = "./docs/ACCENTURE_Luxury_2025.pdf"
convert_pdf_with_descriptions(pdf_file)

print("\nConversion completed!")
