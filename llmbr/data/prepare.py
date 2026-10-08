"""Tokenize cleaned speeches into data/processed/{train,val}.bin.

Run:  python -m llmbr.data.prepare [--config configs/small.toml]

Why .bin files: the whole corpus becomes one long array of token ids
(uint16 — the 32k vocab fits under 65,535), written with numpy. The loader
memory-maps it, so the corpus never needs to fit in RAM, and you can copy just
these two files to the desktop instead of the raw data.

Layout of each file:   speech1 </s> speech2 </s> speech3 </s> ...

The train/val split is done per SPEECH (not per token), with a fixed seed, so
validation text is never seen during training and the split is reproducible.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from llmbr.config import load_config
from llmbr.tokenizer import eot_id, load_tokenizer

BATCH = 1000  # speeches tokenized per encode_batch call (parallel in Rust)


def write_split(texts: list[str], tok, eot: int, path: Path) -> int:
    """Tokenize `texts` and stream them into `path`. Returns the token count."""
    n_tokens = 0
    with open(path, "wb") as f:
        for i in range(0, len(texts), BATCH):
            encs = tok.encode_batch(texts[i:i + BATCH], add_special_tokens=False)
            ids = []
            for e in encs:
                ids.extend(e.ids)
                ids.append(eot)
            np.asarray(ids, dtype=np.uint16).tofile(f)
            n_tokens += len(ids)
    return n_tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/small.toml")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = load_tokenizer(cfg)
    eot = eot_id(tok, cfg)
    assert tok.get_vocab_size() <= 2**16, "vocab too large for uint16 storage"

    src = Path(cfg["paths"]["interim_dir"]) / "speeches.jsonl"
    texts = [json.loads(line)["texto"] for line in open(src, encoding="utf-8")]
    random.Random(cfg["train"]["seed"]).shuffle(texts)
    # At least one validation speech, even on tiny test corpora.
    n_val = max(1, int(len(texts) * args.val_fraction))
    splits = {"val": texts[:n_val], "train": texts[n_val:]}

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": cfg["tokenizer"]["name"], "vocab_size": tok.get_vocab_size(),
            "eot_id": eot, "dtype": "uint16"}
    for name, split_texts in splits.items():
        n = write_split(split_texts, tok, eot, out_dir / f"{name}.bin")
        meta[f"{name}_speeches"] = len(split_texts)
        meta[f"{name}_tokens"] = n
        print(f"{name}: {len(split_texts):,} speeches, {n:,} tokens -> {out_dir / name}.bin")

    # The trainer reads vocab_size and eot_id from here.
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
