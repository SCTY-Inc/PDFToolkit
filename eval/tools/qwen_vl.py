"""Qwen2.5-VL evaluation script."""

from pathlib import Path


def eval_qwen_vl(input_path: Path, output_dir: Path, model_size: str = "7B") -> str:
    """
    Evaluate Qwen2.5-VL for document understanding.

    Install: pip install transformers torch accelerate qwen-vl-utils

    Features:
    - DocVQA score 96.4 (tops among VLMs)
    - Omnidocument parsing (invoices, forms, contracts)
    - Chart/table to JSON/CSV
    - Native dynamic resolution
    """
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from pdf2image import convert_from_path

    output_file = output_dir / f"{input_path.stem}-qwen-vl.md"

    model_id = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Convert PDF to images
    images = convert_from_path(str(input_path))

    results = []
    for i, img in enumerate(images):
        # Save temp image
        temp_path = output_dir / f"_temp_page_{i}.png"
        img.save(temp_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(temp_path)},
                {"type": "text", "text": "Extract all text and describe any charts, tables, or figures. Output in markdown format."}
            ]
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0]

        results.append(f"## Page {i + 1}\n\n{output_text}")
        temp_path.unlink()  # Cleanup

    output_file.write_text("\n\n".join(results), encoding="utf-8")
    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.qwen_vl <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_qwen_vl(input_path, output_dir)
    print(f"Output: {result}")
