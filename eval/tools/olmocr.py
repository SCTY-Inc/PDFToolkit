"""olmOCR evaluation script."""

from pathlib import Path
import subprocess


def eval_olmocr(input_path: Path, output_dir: Path) -> str:
    """
    Evaluate olmOCR for document linearization.

    Install:
        conda create -n olmocr python=3.11
        pip install olmocr[gpu] --extra-index-url https://download.pytorch.org/whl/cu128

    Features:
    - Purpose-built for academic/technical docs
    - 82.4% benchmark score with structured reading order
    - Excellent equation/table handling
    - Auto header/footer removal
    - Requires GPU (RTX 4090, L40S, A100, H100)
    """
    output_file = output_dir / f"{input_path.stem}-olmocr.md"

    # olmOCR is primarily CLI-based
    cmd = [
        "python", "-m", "olmocr.pipeline",
        str(output_dir),
        "--markdown",
        "--pdfs", str(input_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"olmOCR failed: {result.stderr}")

    # Find output file
    for f in output_dir.glob("*.md"):
        if input_path.stem in f.name:
            return str(f)

    return str(output_file)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m eval.tools.olmocr <pdf_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = eval_olmocr(input_path, output_dir)
    print(f"Output: {result}")
