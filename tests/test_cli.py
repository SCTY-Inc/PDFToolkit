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
        assert "benchmark" in result.output
        assert "PDF extraction, analysis, and benchmarking toolkit" in result.output

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

    def test_benchmark_help(self):
        """Benchmark help should show tool and output options."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--tool" in result.output
        assert "-t" in result.output
        assert "--output" in result.output
        assert "docling" in result.output


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

    def test_convert_rejects_describe_without_marker(self, tmp_path):
        """Figure description is only valid for the marker provider."""
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_text("pdf")

        result = runner.invoke(app, ["convert", str(pdf_file), "-p", "docling", "--describe"])
        assert result.exit_code == 1
        assert "--describe is only supported with marker" in result.output


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


class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_missing_file(self, tmp_path):
        """Benchmark should error on missing file."""
        result = runner.invoke(app, ["benchmark", str(tmp_path / "nonexistent.pdf")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_benchmark_invalid_tool(self, tmp_path):
        """Benchmark should reject unknown tool names."""
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_text("pdf")

        result = runner.invoke(app, ["benchmark", str(pdf_file), "-t", "unknown"])
        assert result.exit_code == 1
        assert "Unknown benchmark tool" in result.output

    def test_benchmark_deduplicates_tool_names(self, tmp_path, monkeypatch):
        """Repeated tool names should only run once."""
        from pdftoolkit import cli

        class DummyResult:
            def __init__(self, tool: str):
                self.tool = tool
                self.success = True
                self.error = None
                self.time_seconds = 0.01
                self.output_size_bytes = 1
                self.commercial_use = "yes"

        captured = {}

        def fake_run_benchmark(input_path, output_dir, tool_names, registry):
            captured["tool_names"] = tool_names
            return [DummyResult(name) for name in tool_names]

        monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
        monkeypatch.setattr(cli, "save_results", lambda results, path: None)

        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_text("pdf")

        result = runner.invoke(app, ["benchmark", str(pdf_file), "-t", "docling", "-t", "docling"])
        assert result.exit_code == 0
        assert captured["tool_names"] == ["docling"]
