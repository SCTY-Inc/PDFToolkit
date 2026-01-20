"""InternVL2.5 evaluation script."""

from pathlib import Path


def eval_internvl(input_path: Path, output_dir: Path, model_size: str = "8B") -> str:
    """
    Evaluate InternVL2.5 for document understanding.

    Install: uv pip install transformers torch flash-attn

    Features:
    - DocVQA 95.1%, InfoVQA 84.1%
    - 4K image support
    - Multi-document analysis
    - Rivals GPT-4o on document tasks
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from pdf2image import convert_from_path

    output_file = output_dir / f"{input_path.stem}-internvl.md"

    model_id = f"OpenGVLab/InternVL2_5-{model_size}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    # Convert PDF to images
    images = convert_from_path(str(input_path))

    results = []
    for i, img in enumerate(images):
        prompt = "<image>\nExtract all text content from this document page. Describe any tables, charts, or figures. Output in markdown format."

        with torch.no_grad():
            outputs = model.generate(
                tokenizer.encode(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=2048,
                do_sample=False,
                images=[img],
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(f"## Page {i + 1}\n\n{response}")

    output_file.write_text("\n\n".join(results), encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.internvl <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_internvl(input_path, output_dir)
    print(f"Output: {result}")
