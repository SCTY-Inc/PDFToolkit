"""Tests for pdftoolkit.benchmark."""

from pathlib import Path

from pdftoolkit.benchmark import (
    ToolSpec,
    get_default_benchmark_tools,
    run_benchmark,
    run_tool,
)


class TestBenchmarkDefaults:
    """Tests for default benchmark tool selection."""

    def test_default_tools_are_commercial_and_low_friction(self):
        """Benchmark should default to the commercial-safe local baseline."""
        assert get_default_benchmark_tools(env={}) == ["docling"]

    def test_default_tools_expand_when_api_keys_are_present(self):
        """Cloud/API-backed defaults should opt in when credentials are available."""
        env = {
            "OPENAI_API_KEY": "test-openai",
            "MISTRAL_API_KEY": "test-mistral",
        }
        assert get_default_benchmark_tools(env=env) == [
            "docling",
            "markitdown",
            "mistral",
        ]


class TestRunTool:
    """Tests for benchmark execution helpers."""

    def test_run_tool_success(self, tmp_path):
        """run_tool should capture output path, size, and metadata."""

        def runner(input_path: Path, output_dir: Path) -> Path:
            output_file = output_dir / f"{input_path.stem}.md"
            output_file.write_text("hello")
            return output_file

        spec = ToolSpec(
            name="dummy",
            runner=runner,
            description="Dummy tool",
            commercial_use="yes",
            install_notes="",
            default=True,
        )

        input_path = tmp_path / "sample.pdf"
        input_path.write_text("pdf")

        result = run_tool(spec, input_path, tmp_path / "out")

        assert result.tool == "dummy"
        assert result.success is True
        assert result.error is None
        assert result.output_file is not None
        assert result.output_file.endswith("sample.md")
        assert result.output_size_bytes == 5
        assert result.commercial_use == "yes"

    def test_run_tool_failure(self, tmp_path):
        """run_tool should capture exceptions without crashing the benchmark."""

        def runner(input_path: Path, output_dir: Path) -> Path:
            raise RuntimeError("boom")

        spec = ToolSpec(
            name="broken",
            runner=runner,
            description="Broken tool",
            commercial_use="review",
            install_notes="optional dependency",
            default=False,
        )

        input_path = tmp_path / "sample.pdf"
        input_path.write_text("pdf")

        result = run_tool(spec, input_path, tmp_path / "out")

        assert result.tool == "broken"
        assert result.success is False
        assert result.output_file is None
        assert result.output_size_bytes == 0
        assert result.error == "boom"
        assert result.commercial_use == "review"


class TestRunBenchmark:
    """Tests for running multiple benchmark tools."""

    def test_run_benchmark_keeps_requested_order(self, tmp_path):
        """Benchmark results should preserve the user-requested tool order."""

        def make_runner(name: str):
            def runner(input_path: Path, output_dir: Path) -> Path:
                output_file = output_dir / f"{name}.md"
                output_file.write_text(name)
                return output_file

            return runner

        registry = {
            "first": ToolSpec(
                name="first",
                runner=make_runner("first"),
                description="First tool",
                commercial_use="yes",
                install_notes="",
                default=True,
            ),
            "second": ToolSpec(
                name="second",
                runner=make_runner("second"),
                description="Second tool",
                commercial_use="no",
                install_notes="",
                default=False,
            ),
        }

        input_path = tmp_path / "sample.pdf"
        input_path.write_text("pdf")

        results = run_benchmark(
            input_path=input_path,
            output_dir=tmp_path / "bench",
            tool_names=["second", "first"],
            registry=registry,
        )

        assert [result.tool for result in results] == ["second", "first"]
