# LLM-BR — a small GPT trained on Brazilian political speeches

Goal: train a small GPT from scratch (on a desktop GPU or an Apple Silicon Mac)
on public Brazilian political speech: Câmara dos Deputados, Senado Federal and
Presidência da República.

## Layout

```
DIY/
├── configs/            # TOML configs (model size, training hyperparameters, paths)
│   ├── tiny.toml       # ~11M params — laptop MVP
│   └── small.toml      # ~23M params — desktop
├── data/               # NOT in git — everything here can be regenerated
│   ├── raw/            # untouched downloads, one folder per source (JSONL)
│   ├── interim/        # cleaned text, still one record per speech
│   └── processed/      # tokenized train/val binaries the trainer reads
├── checkpoints/        # NOT in git — saved model weights
├── llmbr/              # the Python package
│   ├── config.py       # loads configs/*.toml
│   ├── device.py       # picks cuda / mps / cpu automatically
│   ├── scrape/         # step 1: download raw data
│   │   ├── common.py   # HTTP helper with retries + polite rate limiting
│   │   ├── camara.py   # Câmara dos Deputados API  (working)
│   │   ├── senado.py   # Senado Federal API        (template)
│   │   └── planalto.py # Presidency speeches        (template)
│   ├── tokenizer/      # pretrained Brazilian Portuguese tokenizer (TucanoBR, 32k vocab)
│   │   └── __init__.py # load_tokenizer(), eot_id()
│   ├── data/
│   │   ├── clean.py    # step 2: strip headers/stage notes, dedupe   (working)
│   │   ├── prepare.py  # step 3: tokenize -> train.bin / val.bin    (working)
│   │   └── loader.py   # GPTDataset + create_dataloader over .bin   (working)
│   ├── model/
│   │   └── gpt.py      # step 4: GPT model                          (working)
│   ├── train.py        # step 5: training loop, checkpoints         (working)
│   └── generate.py     # step 6: sample text from a checkpoint      (working)
└── requirements.txt
```

## Setup

Python 3.11+ (uses `tomllib`). One virtual environment per machine, in `DIY/.venv` (gitignored).

**macOS (laptop, Apple GPU via MPS)**
```bash
cd DIY
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (desktop, NVIDIA GPU via CUDA)** — PowerShell:
```powershell
cd DIY
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Install the CUDA build of torch FIRST — plain `pip install torch` on Windows is CPU-only.
# Get the exact command from https://pytorch.org/get-started/locally/ (pip + your CUDA version), e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Windows notes:
- Set `$env:PYTHONUTF8=1` (or permanently: System > Environment Variables) so Portuguese text
  is read/written as UTF-8 everywhere.
- `pip freeze > file` in PowerShell 5 writes UTF-16 (that's why `study/requirements.txt` looks
  garbled on the Mac). Use `pip freeze | Out-File -Encoding utf8 file` instead.
- bf16 mixed precision is used automatically on RTX 30xx and newer; older cards train in float32.
- Every command below is the same on both systems (`python -m llmbr....`).

## Pipeline (run from this `DIY/` folder, venv active)

```bash
# 1. download (resumable — safe to Ctrl+C and re-run)
python -m llmbr.scrape.camara --legislatura 57            # 2023–2027
python -m llmbr.scrape.camara --legislatura 57 --max-deputados 3   # quick test

# 2. clean            python -m llmbr.data.clean
# 3. tokenize/split   python -m llmbr.data.prepare      (downloads the tokenizer once)
#    check batches    python -m llmbr.data.loader
# 4. train            python -m llmbr.train --config configs/tiny.toml    (laptop MVP, ~11M params)
#                     python -m llmbr.train --config configs/small.toml   (desktop, ~23M params)
# 5. generate         python -m llmbr.generate --prompt "Senhor Presidente,"
```

## Câmara legislatures (for `--legislatura`)

| id | years     |
|----|-----------|
| 57 | 2023–2027 |
| 56 | 2019–2023 |
| 55 | 2015–2019 |
| 54 | 2011–2015 |
| 53 | 2007–2011 |
| 52 | 2003–2007 |

## Training on another machine

Code goes through git; data and checkpoints do not. On the desktop: clone, set up the venv
(above), then either re-run the pipeline or just copy `data/processed/` (train.bin, val.bin,
meta.json, tokenizer.json — a few MB per million speeches), then train. `llmbr/device.py`
picks CUDA there automatically. Checkpoints are portable both ways (`best.pt` trained on
Windows can be sampled on the Mac).
