"""Shared utilities for PDFToolkit."""

import base64
from io import BytesIO
from pathlib import Path, PurePosixPath

import torch
from PIL import Image


def get_device() -> str:
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def image_to_base64(image_path: str | Path) -> str:
    """Convert an image file to base64 string."""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()


def image_to_base64_raw(image_path: str | Path) -> str:
    """Convert an image file to base64 string (raw bytes, no re-encoding)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_output_path(output_dir: str | Path, filename: str) -> Path | None:
    """
    Safely resolve an output path, preventing path traversal attacks.

    Returns None if the path would escape the output directory.
    """
    output_dir = Path(output_dir).resolve()
    # Extract just the filename, handling both Unix and Windows paths
    safe_filename = PurePosixPath(filename).name

    if not safe_filename:
        return None

    full_path = (output_dir / safe_filename).resolve()

    if not full_path.is_relative_to(output_dir):
        return None

    return full_path
