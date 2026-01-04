"""GOT-OCR2.0 evaluation script."""

from pathlib import Path


def eval_got_ocr(input_path: Path, output_dir: Path, use_format: bool = True) -> str:
    """
    Evaluate GOT-OCR2.0 for document OCR.

    Install: pip install transformers pdf2image

    Features:
    - Unified OCR for text, math, tables, charts
    - LaTeX/Markdown/TikZ output formats
    - 580M parameter model
    - Multi-page support
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from pdf2image import convert_from_path

    output_file = output_dir / f"{input_path.stem}-got-ocr.md"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForImageTextToText.from_pretrained(
        "stepfun-ai/GOT-OCR-2.0-hf",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

    # Convert PDF to images
    images = convert_from_path(str(input_path))

    results = []
    for i, img in enumerate(images):
        inputs = processor(img, return_tensors="pt", format=use_format).to(device)

        generate_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=4096,
        )

        text = processor.decode(
            generate_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        results.append(f"## Page {i + 1}\n\n{text}")

    output_file.write_text("\n\n".join(results), encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.got_ocr <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_got_ocr(input_path, output_dir)
    print(f"Output: {result}")
