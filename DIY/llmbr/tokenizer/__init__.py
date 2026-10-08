"""Step 3 — the tokenizer.

We use a ready-made Brazilian Portuguese tokenizer instead of training our own:
TucanoBR/Tucano-160m (SentencePiece-style BPE, 32k vocab, trained on GigaVerbo, a
large Brazilian Portuguese corpus). Measured on Câmara speeches:

    gpt2 (English)        2.21 tokens/word   "administração" -> ' administ','ra','ç','ão'
    Tucano-160m           1.27 tokens/word   "administração" -> '▁administração'

~42% fewer tokens = more speech per context window and cheaper training.

The first call downloads tokenizer.json from Hugging Face and saves a local copy
(cfg["paths"]["tokenizer"]), so later runs — and the desktop, once the file is
copied or re-downloaded — work offline.
"""

from pathlib import Path

from tokenizers import Tokenizer


def load_tokenizer(cfg: dict) -> Tokenizer:
    local = Path(cfg["paths"]["tokenizer"])
    if local.exists():
        return Tokenizer.from_file(str(local))
    tok = Tokenizer.from_pretrained(cfg["tokenizer"]["name"])
    local.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(local))
    return tok


def eot_id(tok: Tokenizer, cfg: dict) -> int:
    """Id of the end-of-text token we put between speeches (Tucano: '</s>' = 2)."""
    i = tok.token_to_id(cfg["tokenizer"]["eot_token"])
    if i is None:
        raise ValueError(f"{cfg['tokenizer']['eot_token']!r} is not in the vocabulary")
    return i
