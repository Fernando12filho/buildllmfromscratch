"""Clean raw speeches: data/raw/**/*.jsonl -> data/interim/speeches.jsonl.

Run:  python -m llmbr.data.clean

What gets removed / changed:
  1. The opening speaker header Câmara transcripts start with, e.g.
       "O SR. ABILIO BRUNINI (Bloco/PL - MT. Sem revisão do orador.) - "
     (speaker, party and state are already in the record's metadata).
     Headers of OTHER speakers inside the text, like the session chair's
     "O SR. PRESIDENTE (Arthur Lira. Bloco/PP - AL) - ...", are kept: that is
     real parliamentary dialogue and useful for the model to learn.
  2. Stage notes: "(Palmas.)", "(Risos.)", "(Desligamento do microfone.)", ...
  3. Unicode normalized to NFC; whitespace collapsed (paragraph breaks kept).
  4. Speeches shorter than --min-words dropped (mostly procedural one-liners).
  5. Exact duplicates (same cleaned text) dropped.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

# "O SR. NAME (...) - " or "A SRA. NAME (...) - " at the very start of a speech.
HEADER = re.compile(r"^\s*(?:O SR\.|A SRA\.)\s+[^()]{1,100}\([^()]*\)\s*[-–]\s*")

# Stage notes from the stenographers. Matched by their first word so variants like
# "(Desligamento automático do microfone.)" are caught too.
STAGE_NOTE = re.compile(
    r"\((?:Palmas|Risos|Pausa|Apupos|Vaias|Manifestaç\w+|Tumulto|Desligamento|"
    r"Soa[m]? a campainha|Interrupção|Intervenção fora do microfone|"
    r"SEM REGISTRO TAQUIGRÁFICO|Inaudível)[^()]*\)",
    re.IGNORECASE,
)


def fix_cp1252(text: str) -> str:
    """Some transcripts contain Windows-1252 bytes decoded as Latin-1, leaving
    invisible C1 control chars: '\\x96' should be '–', '\\x93'/'\\x94' should be
    curly quotes, etc. Re-decode just those characters."""
    return re.sub(r"[\x80-\x9f]",
                  lambda m: bytes([ord(m.group())]).decode("cp1252", errors="ignore"),
                  text)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", fix_cp1252(text))
    text = HEADER.sub("", text, count=1)
    text = STAGE_NOTE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)      # runs of spaces -> one space
    text = re.sub(r" *\n *", "\n", text)            # trim spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)          # at most one blank line
    text = re.sub(r" ([,.;:!?])", r"\1", text)      # "palavra ." left by removed notes
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/speeches.jsonl"))
    ap.add_argument("--min-words", type=int, default=50)
    args = ap.parse_args()

    files = sorted(args.raw.glob("**/*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files under {args.raw} — run a scraper first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    stats = Counter()
    words_by_source = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for path in files:
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                stats["read"] += 1
                text = clean_text(rec["texto"])
                n_words = len(text.split())
                if n_words < args.min_words:
                    stats["too_short"] += 1
                    continue
                digest = hashlib.sha1(text.encode()).hexdigest()
                if digest in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(digest)
                rec["texto"] = text
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                words_by_source[rec["source"]] += n_words

    print(f"files: {len(files)}  read: {stats['read']}  kept: {stats['kept']}  "
          f"too short: {stats['too_short']}  duplicates: {stats['duplicate']}")
    for src, w in words_by_source.items():
        print(f"  {src}: {w:,} words")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
