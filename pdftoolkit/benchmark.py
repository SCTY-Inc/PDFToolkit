"""Benchmark helpers for PDF conversion tools."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable


ToolRunner = Callable[[Path, Path], str | Path]


@dataclass(frozen=True)
class ToolSpec:
    """A benchmarkable PDF tool."""

    name: str
    runner: ToolRunner
    description: str
    commercial_use: str
    install_notes: str
    default: bool = False
    required_env: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class BenchmarkResult:
    """Result of running one benchmark tool on one document."""

    tool: str
    description: str
    commercial_use: str
    input_file: str
    output_file: str | None
    success: bool
    error: str | None
    time_seconds: float
    output_size_bytes: int
    install_notes: str


def _run_docling(input_path: Path, output_dir: Path) -> Path:
    from pdftoolkit.providers import convert_docling

    return convert_docling(input_path, output_dir)


def _run_marker(input_path: Path, output_dir: Path) -> Path:
    from pdftoolkit.providers import convert_marker

    return convert_marker(input_path, output_dir, describe=False)


def _run_megaparse(input_path: Path, output_dir: Path) -> Path:
    from pdftoolkit.providers import convert_megaparse

    return convert_megaparse(input_path, output_dir)


def _run_markitdown(input_path: Path, output_dir: Path) -> Path:
    from pdftoolkit.providers import convert_markitdown

    return convert_markitdown(input_path, output_dir)


def _run_mistral(input_path: Path, output_dir: Path) -> Path:
    from pdftoolkit.providers import convert_mistral

    return convert_mistral(input_path, output_dir)


def _run_mineru(input_path: Path, output_dir: Path) -> str:
    from eval.tools.mineru import eval_mineru

    return eval_mineru(input_path, output_dir)


def _run_olmocr(input_path: Path, output_dir: Path) -> str:
    from eval.tools.olmocr import eval_olmocr

    return eval_olmocr(input_path, output_dir)


def _run_paddleocr(input_path: Path, output_dir: Path) -> str:
    from eval.tools.paddleocr import eval_paddleocr

    return eval_paddleocr(input_path, output_dir)


def _run_got_ocr(input_path: Path, output_dir: Path) -> str:
    from eval.tools.got_ocr import eval_got_ocr

    return eval_got_ocr(input_path, output_dir)


def _run_qwen_vl(input_path: Path, output_dir: Path) -> str:
    from eval.tools.qwen_vl import eval_qwen_vl

    return eval_qwen_vl(input_path, output_dir)


def _run_internvl(input_path: Path, output_dir: Path) -> str:
    from eval.tools.internvl import eval_internvl

    return eval_internvl(input_path, output_dir)


def _run_nanonets(input_path: Path, output_dir: Path) -> str:
    from eval.tools.nanonets import eval_nanonets

    return eval_nanonets(input_path, output_dir)


def get_tool_registry() -> dict[str, ToolSpec]:
    """Return all known benchmark tools."""
    return {
        "docling": ToolSpec(
            name="docling",
            runner=_run_docling,
            description="IBM Docling markdown conversion",
            commercial_use="yes",
            install_notes="Included by default",
            default=True,
        ),
        "marker": ToolSpec(
            name="marker",
            runner=_run_marker,
            description="Marker PDF conversion with layout preservation",
            commercial_use="review",
            install_notes="GPL code and restricted model weights",
        ),
        "megaparse": ToolSpec(
            name="megaparse",
            runner=_run_megaparse,
            description="MegaParse structural parsing",
            commercial_use="yes",
            install_notes="Optional install: megaparse + unstructured extras",
        ),
        "markitdown": ToolSpec(
            name="markitdown",
            runner=_run_markitdown,
            description="Microsoft MarkItDown conversion",
            commercial_use="yes",
            install_notes="Requires OPENAI_API_KEY",
            default=True,
            required_env=("OPENAI_API_KEY",),
        ),
        "mistral": ToolSpec(
            name="mistral",
            runner=_run_mistral,
            description="Mistral OCR API markdown extraction",
            commercial_use="yes",
            install_notes="Requires MISTRAL_API_KEY",
            default=True,
            required_env=("MISTRAL_API_KEY",),
        ),
        "mineru": ToolSpec(
            name="mineru",
            runner=_run_mineru,
            description="MinerU PDF parsing",
            commercial_use="no",
            install_notes="AGPL; optional install: mineru[all] or magic-pdf",
        ),
        "olmocr": ToolSpec(
            name="olmocr",
            runner=_run_olmocr,
            description="olmOCR document linearization",
            commercial_use="yes",
            install_notes="Optional install; GPU-oriented",
        ),
        "paddleocr": ToolSpec(
            name="paddleocr",
            runner=_run_paddleocr,
            description="PaddleOCR PP-Structure document parsing",
            commercial_use="yes",
            install_notes="Optional install: paddleocr[doc-parser]",
        ),
        "got-ocr": ToolSpec(
            name="got-ocr",
            runner=_run_got_ocr,
            description="GOT-OCR2.0 multimodal OCR",
            commercial_use="review",
            install_notes="Optional install; verify model license before production use",
        ),
        "qwen-vl": ToolSpec(
            name="qwen-vl",
            runner=_run_qwen_vl,
            description="Qwen2.5-VL document understanding",
            commercial_use="review",
            install_notes="Optional install; verify model license before production use",
        ),
        "internvl": ToolSpec(
            name="internvl",
            runner=_run_internvl,
            description="InternVL document understanding",
            commercial_use="review",
            install_notes="Optional install; verify model license before production use",
        ),
        "nanonets": ToolSpec(
            name="nanonets",
            runner=_run_nanonets,
            description="Nanonets OCR2 extraction",
            commercial_use="review",
            install_notes="Optional install; verify model license before production use",
        ),
    }


def _has_required_env(spec: ToolSpec, env: Mapping[str, str]) -> bool:
    """Return True when the tool's required environment variables are present."""
    return all(env.get(name) for name in spec.required_env)


def get_default_benchmark_tools(
    registry: dict[str, ToolSpec] | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the default benchmark tool order."""
    registry = registry or get_tool_registry()
    env = os.environ if env is None else env

    return [
        name
        for name, spec in registry.items()
        if spec.default and _has_required_env(spec, env)
    ]


def run_tool(spec: ToolSpec, input_path: Path, output_dir: Path) -> BenchmarkResult:
    """Run a single tool and capture timing and output metadata."""
    tool_output_dir = output_dir / spec.name
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    output_file: str | None = None
    success = False
    error: str | None = None
    output_size = 0

    try:
        output_path = Path(spec.runner(input_path, tool_output_dir))
        output_file = str(output_path)
        if output_path.exists():
            output_size = output_path.stat().st_size
            success = True
        else:
            error = f"Output file not found: {output_path}"
    except Exception as exc:  # pragma: no cover - exercised via tests
        error = str(exc)

    elapsed = perf_counter() - start

    return BenchmarkResult(
        tool=spec.name,
        description=spec.description,
        commercial_use=spec.commercial_use,
        input_file=str(input_path),
        output_file=output_file,
        success=success,
        error=error,
        time_seconds=round(elapsed, 2),
        output_size_bytes=output_size,
        install_notes=spec.install_notes,
    )


def run_benchmark(
    input_path: Path,
    output_dir: Path,
    tool_names: list[str],
    registry: dict[str, ToolSpec] | None = None,
) -> list[BenchmarkResult]:
    """Run a benchmark over the requested tools."""
    registry = registry or get_tool_registry()
    results = []

    for name in tool_names:
        results.append(run_tool(registry[name], input_path, output_dir))

    return results


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Persist benchmark results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
