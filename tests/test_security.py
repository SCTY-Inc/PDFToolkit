"""Security regression tests for PDFToolkit."""

import tempfile
from pathlib import Path

import pytest

from pdftoolkit.utils import safe_output_path


class TestPathTraversal:
    """Regression tests for path traversal vulnerability fix."""

    def test_simple_filename(self):
        """Normal filename should work."""
        result = safe_output_path("/output", "image.jpg")
        assert result == Path("/output/image.jpg")

    def test_path_traversal_dotdot(self):
        """../../../etc/passwd should be blocked."""
        result = safe_output_path("/output", "../../../etc/passwd")
        # Should return just "passwd" in output dir, not escape
        assert result == Path("/output/passwd")

    def test_path_traversal_nested(self):
        """Nested path components should be stripped."""
        result = safe_output_path("/output", "foo/bar/../../../etc/passwd")
        assert result == Path("/output/passwd")

    def test_absolute_path(self):
        """Absolute paths should be stripped to basename."""
        result = safe_output_path("/output", "/etc/passwd")
        assert result == Path("/output/passwd")

    def test_empty_filename(self):
        """Empty filename after sanitization should return None."""
        result = safe_output_path("/output", "../../../")
        assert result is None

    def test_dotfile(self):
        """Dotfiles should work (but be stripped to basename)."""
        result = safe_output_path("/output", ".hidden")
        assert result == Path("/output/.hidden")

    def test_with_real_tempdir(self):
        """Integration test with real temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_resolved = Path(tmpdir).resolve()

            # This should work
            result = safe_output_path(tmpdir, "test.jpg")
            assert result is not None
            assert result.is_relative_to(tmpdir_resolved)

            # This should be sanitized
            result = safe_output_path(tmpdir, "../../../etc/passwd")
            assert result is not None
            assert result.is_relative_to(tmpdir_resolved)
            assert result.name == "passwd"

            # This should fail
            result = safe_output_path(tmpdir, "")
            assert result is None
