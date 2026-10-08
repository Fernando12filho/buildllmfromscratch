# LLM-BR — a small GPT trained on Brazilian political speeches

Goal: train a small GPT from scratch (on a desktop GPU or an Apple Silicon Mac)
on public Brazilian political speech: Câmara dos Deputados, Senado Federal and
Presidência da República.

## Layout

```
DIY/
├── configs/            # TOML configs (model size, training hyperparameters, paths)
│   └── small.toml
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
│   ├── data/           # step 2: clean + split
│   │   ├── clean.py    # (template)
│   │   └── prepare.py  # tokenize -> train.bin / val.bin (template)
│   ├── tokenizer/      # step 3: train a Portuguese BPE tokenizer
│   │   └── train_tokenizer.py (template)
│   ├── model/          # step 4: the GPT itself
│   │   └── gpt.py      # (template — port ../gpt2.py here)
│   ├── train.py        # step 5: training loop (template)
│   └── generate.py     # step 6: sample text from a checkpoint (template)
└── requirements.txt
```

## Pipeline (run from this `DIY/` folder)

```bash
# 1. download (resumable — safe to Ctrl+C and re-run)
python -m llmbr.scrape.camara --legislatura 57            # 2023–2027
python -m llmbr.scrape.camara --legislatura 57 --max-deputados 3   # quick test

# 2. clean            python -m llmbr.data.clean
# 3. tokenizer        python -m llmbr.tokenizer.train_tokenizer
# 4. tokenize/split   python -m llmbr.data.prepare
# 5. train            python -m llmbr.train --config configs/small.toml
# 6. generate         python -m llmbr.generate --prompt "Senhor Presidente,"
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

Code goes through git; data does not. On the desktop: clone, `pip install -r requirements.txt`,
re-run the scrapers (or copy `data/` over), then train. `llmbr/device.py` picks CUDA there automatically.
