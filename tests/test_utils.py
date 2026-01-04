"""Tests for pdftoolkit.utils module."""

import tempfile
from pathlib import Path

import pytest

from pdftoolkit.utils import get_device, image_to_base64, image_to_base64_raw


class TestGetDevice:
    """Tests for device detection."""

    def test_returns_valid_device(self):
        """Should return mps, cuda, or cpu."""
        device = get_device()
        assert device in ("mps", "cuda", "cpu")


class TestImageToBase64:
    """Tests for base64 encoding functions."""

    def test_image_to_base64_with_valid_image(self, tmp_path):
        """Should encode a valid image to base64."""
        # Create a minimal valid JPEG
        from PIL import Image

        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (10, 10), color="red")
        img.save(img_path, "JPEG")

        result = image_to_base64(img_path)
        assert isinstance(result, str)
        assert len(result) > 0
        # Base64 should be decodable
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_image_to_base64_raw_with_valid_file(self, tmp_path):
        """Should encode raw file bytes to base64."""
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(b"test data")

        result = image_to_base64_raw(file_path)
        assert isinstance(result, str)

        import base64

        decoded = base64.b64decode(result)
        assert decoded == b"test data"
