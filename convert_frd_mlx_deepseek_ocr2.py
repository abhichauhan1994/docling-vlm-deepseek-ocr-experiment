"""PDF → page images → MLX DeepSeek-OCR-2 → markdown (no Docling VlmPipeline)."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=False)

ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "data" / "input" / "FRD_Phase1.pdf"
DEFAULT_OUT = ROOT / "data" / "output" / "mlx_deepseek_ocr2_bf16"
MODEL_8BIT = "mlx-community/DeepSeek-OCR-2-8bit"
MODEL_BF16 = "mlx-community/DeepSeek-OCR-2-bf16"

# DeepSeek-OCR-2 expects a SHORT grounding prompt. Long "Requirements:" lists
# make it continue the instructions instead of reading the page image
# (bf16 custom-prompt run produced meta-rules garbage, no document text).
DEFAULT_PROMPT = "<|grounding|>Convert the document to markdown."


def _strip_grounding(text: str) -> str:
    """Remove DeepSeek ref/det tokens for a readable markdown view."""
    text = re.sub(
        r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>\s*",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip() + "\n"


@app.command()
def main(
    pdf: Path = typer.Option(DEFAULT_PDF, exists=True, readable=True),
    out_dir: Path = typer.Option(DEFAULT_OUT),
    model_id: Optional[str] = typer.Option(
        None,
        help="HF MLX repo override (default: bf16)",
    ),
    use_8bit: bool = typer.Option(
        False,
        "--8bit",
        help="Use 8bit weights instead of default bf16",
    ),
    prompt: Optional[str] = typer.Option(
        None,
        help="Override conversion prompt (still should start with <|grounding|>)",
    ),
    max_pages: Optional[int] = typer.Option(None, help="Only first N pages"),
    page: Optional[int] = typer.Option(
        None, help="1-based single page (overrides max_pages)"
    ),
    max_tokens: int = typer.Option(
        8192, help="Max new tokens per page (dense FR tables need headroom)"
    ),
    scale: float = typer.Option(2.0, help="PDF render scale"),
    skip_existing: bool = typer.Option(
        False,
        help="Skip pages that already have raw_page_XXXX.txt",
    ),
) -> None:
    import pypdfium2 as pdfium
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    resolved_model = (
        model_id or (MODEL_8BIT if use_8bit else MODEL_BF16)
    )
    active_prompt = (prompt or DEFAULT_PROMPT).strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    # trust_remote_code=False: HF remote modeling files break under current
    # transformers; mlx-vlm then loads native DeepseekOCR2Processor.
    typer.echo(f"Loading {resolved_model} …")
    t_load = time.perf_counter()
    model, processor = load(resolved_model, trust_remote_code=False)
    formatted_prompt = apply_chat_template(
        processor, model.config, active_prompt, num_images=1
    )
    typer.echo(f"Model loaded in {time.perf_counter() - t_load:.1f}s")
    typer.echo(f"Prompt ({len(active_prompt)} chars): {active_prompt[:120]}…")

    doc = pdfium.PdfDocument(str(pdf))
    n_total = len(doc)
    if page is not None:
        if page < 1 or page > n_total:
            raise typer.BadParameter(f"page must be 1..{n_total}")
        indices = [page - 1]
    else:
        n = n_total if max_pages is None else min(max_pages, n_total)
        indices = list(range(n))

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    page_meta: list[dict] = []
    combined_raw: list[str] = []
    combined_clean: list[str] = []

    for i in indices:
        page_no = i + 1
        raw_path = pages_dir / f"raw_page_{page_no:04d}.txt"
        img_path = pages_dir / f"page_{page_no:04d}.png"
        clean_path = pages_dir / f"clean_page_{page_no:04d}.md"

        if skip_existing and raw_path.exists() and raw_path.stat().st_size > 0:
            text = raw_path.read_text(encoding="utf-8")
            typer.echo(f"page {page_no}/{n_total}: skip (cached)")
        else:
            pil = doc[i].render(scale=scale).to_pil()
            pil.save(img_path)
            t_page = time.perf_counter()
            typer.echo(f"page {page_no}/{n_total}: generating …")
            result = generate(
                model,
                processor,
                prompt=formatted_prompt,
                image=str(img_path),
                max_tokens=max_tokens,
                temperature=0.0,
                verbose=False,
            )
            text = getattr(result, "text", None) or str(result)
            raw_path.write_text(text, encoding="utf-8")
            elapsed_page = time.perf_counter() - t_page
            page_meta.append(
                {
                    "page": page_no,
                    "elapsed_sec": round(elapsed_page, 2),
                    "chars": len(text),
                    "has_ref": "<|ref|>" in text,
                    "has_html_table": "<table" in text.lower(),
                    "md_pipe_rows": sum(
                        1 for ln in text.splitlines() if ln.strip().startswith("|")
                    ),
                }
            )
            typer.echo(
                f"page {page_no}: {elapsed_page:.1f}s chars={len(text)} "
                f"ref={page_meta[-1]['has_ref']} "
                f"html_table={page_meta[-1]['has_html_table']}"
            )

        clean = _strip_grounding(text)
        clean_path.write_text(clean, encoding="utf-8")
        combined_raw.append(f"\n\n<!-- page {page_no} -->\n\n{text.strip()}\n")
        combined_clean.append(f"\n\n<!-- page {page_no} -->\n\n{clean.strip()}\n")

    stem = pdf.stem
    if page is not None:
        suffix = f".page{page}"
    elif max_pages is not None:
        suffix = f".first{max_pages}"
    else:
        suffix = ""

    raw_md = out_dir / f"{stem}{suffix}.deepseek_ocr2.raw.md"
    clean_md = out_dir / f"{stem}{suffix}.deepseek_ocr2.md"
    raw_md.write_text("".join(combined_raw).lstrip() + "\n", encoding="utf-8")
    clean_md.write_text("".join(combined_clean).lstrip() + "\n", encoding="utf-8")

    meta = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "pdf": str(pdf),
        "model_id": resolved_model,
        "precision": "8bit" if use_8bit else "bf16",
        "prompt": active_prompt,
        "max_tokens": max_tokens,
        "scale": scale,
        "pages": [i + 1 for i in indices],
        "page_meta": page_meta,
        "outputs": {"raw_markdown": str(raw_md), "clean_markdown": str(clean_md)},
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    typer.echo(
        json.dumps(
            {"ok": True, "clean_md": str(clean_md), "elapsed_sec": meta["elapsed_sec"]}
        )
    )


if __name__ == "__main__":
    app()
