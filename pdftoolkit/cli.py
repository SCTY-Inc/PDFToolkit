"""CLI interface for PDFToolkit using Typer."""

from enum import Enum
from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(
    help="PDF extraction and analysis toolkit",
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


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input PDF file"),
    provider: ConvertProvider = typer.Option(ConvertProvider.docling, "-p", "--provider", help="Conversion provider"),
    output: Path = typer.Option(Path("output"), "-o", "--output", help="Output directory"),
    describe: bool = typer.Option(False, "--describe", help="Add AI image descriptions (marker only)"),
) -> None:
    """Convert PDF to markdown."""
    if not input_path.exists():
        rprint(f"[red]Error:[/red] File not found: {input_path}")
        raise typer.Exit(1)

    if not input_path.suffix.lower() == ".pdf":
        rprint(f"[yellow]Warning:[/yellow] Input file may not be a PDF: {input_path}")

    from pdftoolkit.providers import (
        convert_docling,
        convert_marker,
        convert_megaparse,
        convert_markitdown,
        convert_mistral,
    )

    rprint(f"Converting [cyan]{input_path}[/cyan] with [green]{provider.value}[/green]...")

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

    rprint(f"[green]Saved:[/green] {output_path}")


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., help="Input image file or directory"),
    provider: AnalyzeProvider = typer.Option(AnalyzeProvider.ollama, "-p", "--provider", help="Analysis provider"),
    query: str = typer.Option("Describe this visualization", "-q", "--query", help="Analysis query/prompt"),
    threshold: float = typer.Option(0.5, "--threshold", help="Confidence threshold for chart detection"),
) -> None:
    """Analyze images/charts with vision models."""
    if not input_path.exists():
        rprint(f"[red]Error:[/red] Path not found: {input_path}")
        raise typer.Exit(1)

    from pdftoolkit.providers import (
        analyze_ollama,
        analyze_together,
        analyze_colqwen,
    )

    # Collect files to process
    if input_path.is_dir():
        files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))
        if not files:
            rprint(f"[yellow]No image files found in:[/yellow] {input_path}")
            raise typer.Exit(1)
    else:
        files = [input_path]

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
                default_queries = [
                    "chart showing numeric trends",
                    "graph with multiple data points",
                    "statistical visualization",
                    "data showing growth over time",
                    "comparison between different values",
                    "percentage or ratio data",
                ]
                queries = [query] if query != "Describe this visualization" else default_queries
                results = analyze_colqwen(file_path, queries, threshold=threshold)

                if results:
                    rprint("[bold]Relevant content found:[/bold]")
                    for q, score in results:
                        rprint(f"  [green]{score:.3f}[/green] {q}")
                else:
                    rprint(f"[yellow]No matches above threshold ({threshold})[/yellow]")


if __name__ == "__main__":
    app()
