"""CLI interface for PDFToolkit using Typer."""

import os
from enum import Enum
from pathlib import Path

import typer
from rich import print as rprint

from pdftoolkit.benchmark import (
    get_default_benchmark_tools,
    get_tool_registry,
    run_benchmark,
    save_results,
)

app = typer.Typer(
    help="PDF extraction, analysis, and benchmarking toolkit",
    no_args_is_help=True,
)


class ConvertProvider(str, Enum):
    """Available PDF conversion providers."""
    docling = "docling"
    marker = "marker"
    megaparse = "megaparse"
    markitdown = "markitdown"
    mistral = "mistral"


class AnalyzeProvider(str, Enum):
    """Available image analysis providers."""
    ollama = "ollama"
    together = "together"
    colqwen = "colqwen"


# Default queries for ColQwen relevance scoring
COLQWEN_DEFAULT_QUERIES = [
    "chart showing numeric trends",
    "graph with multiple data points",
    "statistical visualization",
    "data showing growth over time",
    "comparison between different values",
    "percentage or ratio data",
]

# Supported image extensions (lowercase)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BENCHMARK_TOOL_NAMES = ", ".join(get_tool_registry())


def _error(message: str) -> None:
    """Print an error and exit."""
    rprint(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


def _warn(message: str) -> None:
    """Print a warning."""
    rprint(f"[yellow]Warning:[/yellow] {message}")


def _ensure_exists(path: Path, label: str = "File") -> None:
    """Exit if a required path does not exist."""
    if not path.exists():
        _error(f"{label} not found: {path}")


def _warn_if_not_pdf(path: Path) -> None:
    """Warn when a path does not look like a PDF."""
    if path.suffix.lower() != ".pdf":
        _warn(f"Input file may not be a PDF: {path}")


def _require_env_vars(*names: str) -> None:
    """Exit when required environment variables are missing."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        _error(f"Missing required environment variable(s): {', '.join(missing)}")


def _collect_image_files(input_path: Path) -> list[Path]:
    """Collect image files from a path or directory."""
    if input_path.is_dir():
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            _error(f"No image files found in: {input_path}")
        return files

    return [input_path]


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input PDF file"),
    provider: ConvertProvider = typer.Option(ConvertProvider.docling, "-p", "--provider", help="Conversion provider"),
    output: Path = typer.Option(Path("output"), "-o", "--output", help="Output directory"),
    describe: bool = typer.Option(False, "--describe", help="Add AI image descriptions (marker only)"),
) -> None:
    """Convert PDF to markdown."""
    _ensure_exists(input_path)
    _warn_if_not_pdf(input_path)

    if describe and provider is not ConvertProvider.marker:
        _error("--describe is only supported with marker")

    if provider is ConvertProvider.markitdown:
        _require_env_vars("OPENAI_API_KEY")
    elif provider is ConvertProvider.mistral:
        _require_env_vars("MISTRAL_API_KEY")

    from pdftoolkit.providers import (
        convert_docling,
        convert_marker,
        convert_megaparse,
        convert_markitdown,
        convert_mistral,
    )

    rprint(f"Converting [cyan]{input_path}[/cyan] with [green]{provider.value}[/green]...")

    try:
        match provider:
            case ConvertProvider.docling:
                output_path = convert_docling(input_path, output)
            case ConvertProvider.marker:
                output_path = convert_marker(input_path, output, describe=describe)
            case ConvertProvider.megaparse:
                output_path = convert_megaparse(input_path, output)
            case ConvertProvider.markitdown:
                output_path = convert_markitdown(input_path, output)
            case ConvertProvider.mistral:
                output_path = convert_mistral(input_path, output)
    except ImportError as exc:
        _error(f"Provider '{provider.value}' is not available: {exc}")

    rprint(f"[green]Saved:[/green] {output_path}")


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., help="Input image file or directory"),
    provider: AnalyzeProvider = typer.Option(AnalyzeProvider.ollama, "-p", "--provider", help="Analysis provider"),
    query: str = typer.Option("Describe this visualization", "-q", "--query", help="Analysis query/prompt"),
    threshold: float = typer.Option(0.5, "--threshold", help="Confidence threshold for chart detection"),
) -> None:
    """Analyze images/charts with vision models."""
    _ensure_exists(input_path, label="Path")

    if provider is AnalyzeProvider.together:
        _require_env_vars("TOGETHER_API_KEY")

    from pdftoolkit.providers import (
        analyze_ollama,
        analyze_together,
        analyze_colqwen,
    )

    files = _collect_image_files(input_path)

    for file_path in files:
        rprint(f"\nAnalyzing [cyan]{file_path}[/cyan] with [green]{provider.value}[/green]...")

        match provider:
            case AnalyzeProvider.ollama:
                result = analyze_ollama(file_path, query)
                rprint(f"[bold]Result:[/bold]\n{result}")

            case AnalyzeProvider.together:
                result = analyze_together(file_path, query, threshold=threshold)
                rprint(f"[bold]Result:[/bold]\n{result}")

            case AnalyzeProvider.colqwen:
                # ColQwen uses multiple queries for relevance scoring
                queries = [query] if query != "Describe this visualization" else COLQWEN_DEFAULT_QUERIES
                results = analyze_colqwen(file_path, queries, threshold=threshold)

                if results:
                    rprint("[bold]Relevant content found:[/bold]")
                    for q, score in results:
                        rprint(f"  [green]{score:.3f}[/green] {q}")
                else:
                    rprint(f"[yellow]No matches above threshold ({threshold})[/yellow]")


@app.command()
def benchmark(
    input_path: Path = typer.Argument(..., help="Input PDF file"),
    tool: list[str] | None = typer.Option(
        None,
        "-t",
        "--tool",
        help=f"Benchmark tool to run. Repeat for multiple tools. Available: {BENCHMARK_TOOL_NAMES}",
    ),
    output: Path = typer.Option(Path("output/benchmark"), "-o", "--output", help="Benchmark output directory"),
) -> None:
    """Benchmark one PDF across multiple extraction tools."""
    _ensure_exists(input_path)
    _warn_if_not_pdf(input_path)

    registry = get_tool_registry()
    selected_tools = list(dict.fromkeys(tool or get_default_benchmark_tools(registry=registry)))

    unknown_tools = [name for name in selected_tools if name not in registry]
    if unknown_tools:
        _error(f"Unknown benchmark tool(s): {', '.join(unknown_tools)}")

    if not selected_tools:
        _error("No benchmark tools are runnable by default. Set API keys or pass --tool explicitly.")

    run_dir = output / input_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    rprint(f"Benchmarking [cyan]{input_path}[/cyan] with: [green]{', '.join(selected_tools)}[/green]")
    results = run_benchmark(input_path=input_path, output_dir=run_dir, tool_names=selected_tools, registry=registry)

    results_path = run_dir / "results.json"
    save_results(results, results_path)

    rprint("\n[bold]Summary:[/bold]")
    for result in results:
        status = "OK" if result.success else "FAIL"
        size = f"{result.output_size_bytes:,} bytes" if result.success else "-"
        commercial = result.commercial_use.upper()
        if result.success:
            rprint(
                f"  [green]{result.tool}[/green] [{commercial}] {status} "
                f"({result.time_seconds:.2f}s, {size})"
            )
        else:
            rprint(
                f"  [red]{result.tool}[/red] [{commercial}] {status} "
                f"({result.time_seconds:.2f}s) - {result.error}"
            )

    rprint(f"\n[green]Results saved:[/green] {results_path}")

    if not any(result.success for result in results):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
