"""TEMPLATE — train a byte-level BPE tokenizer on data/interim/speeches.jsonl.

Why not GPT-2's tokenizer: it was trained on English, so Portuguese words
("administração", "parlamentares") get split into many tiny pieces. A tokenizer
trained on our corpus gives ~30–40% fewer tokens for the same text, which means
more text fits in the context window and training is cheaper.

Steps to implement (Hugging Face `tokenizers` library):
  1. from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
  2. tok = Tokenizer(models.BPE())
     tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
     tok.decoder = decoders.ByteLevel()
  3. trainer = trainers.BpeTrainer(vocab_size=cfg["tokenizer"]["vocab_size"],
                                   special_tokens=["<|eot|>"],
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
  4. tok.train_from_iterator(<generator yielding each speech's texto>, trainer)
  5. tok.save(cfg["paths"]["tokenizer"])
  6. Sanity check: encode/decode a sample sentence and print the pieces.

Your ../bpetokenizer.py is a from-scratch BPE — worth comparing against this.
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.toml")
    args = ap.parse_args()
    # TODO: steps 1–6 above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
