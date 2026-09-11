# Docling VLM + DeepSeek-OCR experiment

Isolated from `docling-pdf-healer`. Converts `FRD_Phase1.pdf` with Docling's
**`VlmPipeline`** and the **`deepseek_ocr`** preset.

## Important (Apple Silicon)

Docling does **not** ship a native MLX engine for DeepSeek-OCR. The
`deepseek_ocr` preset only supports:

- `api_ollama` (default) → `http://localhost:11434`
- `api_lmstudio`


## Setup

```bash
cd /Users/abhishek/Projects/docling-vlm-deepseek-ocr-experiment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# ensure Ollama is up and the model is pulled
ollama list | grep deepseek-ocr
```

## Run

```bash
# smoke (first 2 pages)
python convert_frd.py --max-pages 2

# full 48-page FRD (slow: VLM page-by-page)
python convert_frd.py
```

Outputs land in `data/output/`:

- `FRD_Phase1.vlm.md` — markdown
- `FRD_Phase1.vlm.json` — DoclingDocument JSON
- `run_meta.json` — timing / options


## Option 2: MLX DeepSeek-OCR-2 (Apple Silicon)

Bypasses Docling VlmPipeline / Ollama. Renders PDF pages and runs
**`mlx-community/DeepSeek-OCR-2-bf16`** in-process via `mlx_vlm`
(`trust_remote_code=False`). Custom grounding prompt targets FRD table/list issues.

```bash
source .venv/bin/activate
pip install -U mlx-vlm pypdfium2

# smoke one table-heavy page
python convert_frd_mlx_deepseek_ocr2.py --page 13

# full document (bf16 default; outputs under mlx_deepseek_ocr2_bf16/)
python convert_frd_mlx_deepseek_ocr2.py

# optional: old 8bit weights
python convert_frd_mlx_deepseek_ocr2.py --8bit --out-dir data/output/mlx_deepseek_ocr2_8bit
```

Outputs (bf16):

- `data/output/mlx_deepseek_ocr2_bf16/FRD_Phase1.deepseek_ocr2.md` — grounding tags stripped
- `…/FRD_Phase1.deepseek_ocr2.raw.md` — raw (`<|ref|>` / HTML tables)
- `…/run_meta.json` — model, prompt, per-page timing

Retired 8bit run (if present): `data/output/mlx_deepseek_ocr2_8bit_retired/`.
