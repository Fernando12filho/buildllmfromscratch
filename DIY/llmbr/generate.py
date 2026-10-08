"""TEMPLATE — sample text from a trained checkpoint.

Run:  python -m llmbr.generate --prompt "Senhor Presidente," --max-new-tokens 200

Steps to implement:
  1. Load checkpoints/last.pt (its saved cfg rebuilds the model) and the tokenizer.
  2. Encode the prompt; loop max_new_tokens times:
       logits = model(idx[:, -context_length:])[:, -1, :] / temperature
       keep only top_k logits, softmax, torch.multinomial -> next token
       stop early on <|eot|>
  3. Decode and print.
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/last.pt")
    ap.add_argument("--prompt", default="Senhor Presidente,")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    # TODO: steps 1–3 above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
