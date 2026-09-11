"""Convert FRD_Phase1 via Docling VlmPipeline + DeepSeek-OCR (Ollama)."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from pydantic import AnyUrl

app = typer.Typer(add_completion=False, no_args_is_help=False)

ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "data" / "input" / "FRD_Phase1.pdf"
DEFAULT_OUT = ROOT / "data" / "output"


def _build_converter(
    *,
    model: str,
    ollama_url: str,
    timeout_sec: float,
    concurrency: int,
    max_tokens: int,
):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    engine = ApiVlmEngineOptions(
        engine_type=VlmEngineType.API_OLLAMA,
        url=AnyUrl(ollama_url),
        timeout=timeout_sec,
        concurrency=concurrency,
        params={"model": model, "max_tokens": max_tokens},
    )
    vlm_options = VlmConvertOptions.from_preset(
        "deepseek_ocr",
        engine_options=engine,
    )
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,  # required for Ollama API
        generate_page_images=True,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


@app.command()
def main(
    pdf: Path = typer.Option(DEFAULT_PDF, exists=True, readable=True, help="Input PDF"),
    out_dir: Path = typer.Option(DEFAULT_OUT, help="Output directory"),
    model: str = typer.Option(
        "deepseek-ocr:latest",
        help="Ollama model tag (preset default deepseek-ocr:3b may not be pulled)",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434/v1/chat/completions",
        help="OpenAI-compatible Ollama chat endpoint",
    ),
    timeout_sec: float = typer.Option(300.0, help="Per-request timeout seconds"),
    concurrency: int = typer.Option(1, help="VLM concurrency (keep 1 on Apple Silicon)"),
    max_tokens: int = typer.Option(8192, help="max_tokens per page"),
    max_pages: Optional[int] = typer.Option(
        None,
        help="If set, only convert the first N pages (writes a temp truncated PDF)",
    ),
) -> None:
    """Run Docling VlmPipeline with deepseek_ocr preset via local Ollama."""
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"host={platform.system()} {platform.machine()} "
        f"model={model} concurrency={concurrency}"
    )
    typer.echo(
        "Note: Docling deepseek_ocr has no MLX engine — uses Ollama API "
        "(Metal acceleration is inside Ollama)."
    )

    src = pdf
    temp_pdf: Path | None = None
    if max_pages is not None:
        if max_pages < 1:
            raise typer.BadParameter("max_pages must be >= 1")
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            typer.echo("Installing pypdf for --max-pages…", err=True)
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
            from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(pdf))
        writer = PdfWriter()
        n = min(max_pages, len(reader.pages))
        for i in range(n):
            writer.add_page(reader.pages[i])
        temp_pdf = out_dir / f"{pdf.stem}.first_{n}_pages.pdf"
        with temp_pdf.open("wb") as fh:
            writer.write(fh)
        src = temp_pdf
        typer.echo(f"Truncated to {n} pages → {src}")

    converter = _build_converter(
        model=model,
        ollama_url=ollama_url,
        timeout_sec=timeout_sec,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    typer.echo(f"Converting {src} …")
    result = converter.convert(src)
    elapsed = time.perf_counter() - t0
    doc = result.document

    stem = pdf.stem + (f".first{max_pages}" if max_pages else "") + ".vlm"
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    meta_path = out_dir / "run_meta.json"

    md = doc.export_to_markdown()
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    meta = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "pdf": str(pdf),
        "source_used": str(src),
        "max_pages": max_pages,
        "model": model,
        "ollama_url": ollama_url,
        "timeout_sec": timeout_sec,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "preset": "deepseek_ocr",
        "engine": "api_ollama",
        "platform": f"{platform.system()}-{platform.machine()}",
        "pages_expected": getattr(result, "input", None)
        and getattr(result.input, "page_count", None),
        "status": str(getattr(result, "status", "")),
        "markdown_chars": len(md),
        "outputs": {"markdown": str(md_path), "json": str(json_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    typer.echo(json.dumps({"ok": True, "elapsed_sec": meta["elapsed_sec"], "md": str(md_path)}))


if __name__ == "__main__":
    app()
