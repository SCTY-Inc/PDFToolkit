"""Tests for pdftoolkit CLI."""

import pytest
from typer.testing import CliRunner

from pdftoolkit.cli import app, ConvertProvider, AnalyzeProvider


runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Main help should show available commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "convert" in result.output
        assert "analyze" in result.output
        assert "PDF extraction and analysis toolkit" in result.output

    def test_convert_help(self):
        """Convert help should show provider options."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "-p" in result.output
        assert "docling" in result.output
        assert "--output" in result.output
        assert "--describe" in result.output

    def test_analyze_help(self):
        """Analyze help should show provider and query options."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "-p" in result.output
        assert "ollama" in result.output
        assert "--query" in result.output
        assert "--threshold" in result.output


class TestProviderEnums:
    """Tests for provider enum validation."""

    def test_convert_providers(self):
        """All expected convert providers should be defined."""
        expected = {"docling", "marker", "megaparse", "markitdown", "mistral"}
        actual = {p.value for p in ConvertProvider}
        assert actual == expected

    def test_analyze_providers(self):
        """All expected analyze providers should be defined."""
        expected = {"ollama", "together", "colqwen"}
        actual = {p.value for p in AnalyzeProvider}
        assert actual == expected

    def test_invalid_convert_provider(self):
        """Invalid provider should fail."""
        result = runner.invoke(app, ["convert", "test.pdf", "-p", "invalid"])
        assert result.exit_code != 0

    def test_invalid_analyze_provider(self):
        """Invalid provider should fail."""
        result = runner.invoke(app, ["analyze", "test.jpg", "-p", "invalid"])
        assert result.exit_code != 0


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_missing_file(self, tmp_path):
        """Convert should error on missing file."""
        result = runner.invoke(app, ["convert", str(tmp_path / "nonexistent.pdf")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_convert_non_pdf_warning(self, tmp_path):
        """Convert should warn on non-PDF files."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")
        result = runner.invoke(app, ["convert", str(txt_file)])
        assert "Warning" in result.output or "may not be a PDF" in result.output


class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_missing_file(self, tmp_path):
        """Analyze should error on missing file."""
        result = runner.invoke(app, ["analyze", str(tmp_path / "nonexistent.jpg")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_analyze_empty_directory(self, tmp_path):
        """Analyze should report no images in empty directory."""
        result = runner.invoke(app, ["analyze", str(tmp_path)])
        assert result.exit_code == 1
        assert "No image files" in result.output
