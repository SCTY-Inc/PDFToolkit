"""Nanonets-OCR2 evaluation script."""

from pathlib import Path


def eval_nanonets(input_path: Path, output_dir: Path) -> str:
    """
    Evaluate Nanonets-OCR2 for enterprise document extraction.

    Install: pip install transformers flash-attn>=2.0.0

    Features:
    - Advanced element recognition (signatures, watermarks, checkboxes)
    - LaTeX equation generation
    - Mermaid code for flowcharts
    - VQA capability with "Not mentioned" responses
    - 85.15% DocVQA accuracy
    - 12+ languages
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pdf2image import convert_from_path

    output_file = output_dir / f"{input_path.stem}-nanonets.md"

    model = AutoModelForImageTextToText.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR2-3B")

    # Convert PDF to images
    images = convert_from_path(str(input_path))

    results = []
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False
        )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(f"## Page {i + 1}\n\n{text}")

    output_file.write_text("\n\n".join(results), encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.nanonets <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_nanonets(input_path, output_dir)
    print(f"Output: {result}")
