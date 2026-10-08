# CLAUDE.md — LLM-BR

A small GPT trained from scratch on public Brazilian political speeches (Câmara dos
Deputados now; Senado and Presidência planned). Python package: `llmbr`. See README.md
for the full pipeline and setup.

## Hard rule: stay inside this directory

**This `DIY/` directory is the whole project. Never create, modify, move or delete
anything outside it** — no edits, no `rm`, no `mv`, no shell redirects (`>`), no
`git checkout`/`git clean` affecting paths outside `DIY/`. That includes the parent repo
folder `build-LLM/`, its `study/` and `agent/` folders, the home directory, and system
or global Python locations (no `pip install` outside `DIY/.venv`).

- Reading files outside is fine when needed (e.g. `../study/gpt2.py` for reference).
- `../study/` is the owner's LLM learning material. It stays in git but is NOT part of the
  project: never edit it, and never import from it.
- Temporary files go in `DIY/` (and get cleaned up) or the session scratchpad, not `/tmp`
  or elsewhere.
- If a task seems to need a change outside `DIY/`, stop and ask first.

## Environment

- Use the project venv only: `DIY/.venv` (macOS: `.venv/bin/python`, Windows:
  `.venv\Scripts\python.exe`). Never install into the global Python.
- Run everything from `DIY/` as modules: `python -m llmbr.<module>`.
- Two machines: MacBook M3 16 GB (MPS, for development and small runs) and a Windows
  desktop with NVIDIA GPU (CUDA, for real training). Code must run on both.

## Pipeline commands

```bash
python -m llmbr.scrape.camara --legislatura 57 [--max-deputados N]   # download (resumable)
python -m llmbr.data.clean                                           # raw -> interim
python -m llmbr.data.prepare --config configs/tiny.toml              # -> train.bin/val.bin
python -m llmbr.data.loader --config configs/tiny.toml               # sanity-check batches
python -m llmbr.train --config configs/tiny.toml [--max-steps 30] [--resume]
python -m llmbr.generate --checkpoint checkpoints/tiny/best.pt --prompt "Senhor Presidente,"
```

`--max-steps 30` is the quick smoke test after changing model/training code.

## Conventions

- **Cross-platform:** `pathlib` for paths; always pass `encoding="utf-8"` to `open()` /
  `read_text()` / `write_text()`; ASCII only in `print()` (Windows consoles/logs use
  cp1252); DataLoader `num_workers=0` by default; device chosen via `llmbr/device.py`.
- **Config, not constants:** model/training sizes live in `configs/*.toml`; `vocab_size` and
  `eot_id` come from `data/processed/meta.json`.
- **Tokenizer:** pretrained `TucanoBR/Tucano-160m` (32k vocab, Brazilian Portuguese);
  token ids stored as `uint16`. Don't switch to a tokenizer with vocab > 65,535 without
  changing the storage dtype.
- **Scrapers:** standard library + certifi only; polite rate limiting via
  `scrape/common.py`; resumable with `.done` files; same JSONL record keys across sources.
- **Comments:** the owner is learning — explain the *why* (shapes, design choices), and
  relate to `../study/` concepts where useful.
- **Never commit** `data/`, `checkpoints/`, `.venv/` (already in `.gitignore`).
- Commit/push only when asked.
