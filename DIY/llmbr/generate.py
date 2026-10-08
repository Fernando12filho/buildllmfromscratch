"""Sample text from a trained checkpoint.

Run:  python -m llmbr.generate --prompt "Senhor Presidente," --max-new-tokens 200
      python -m llmbr.generate --checkpoint checkpoints/best.pt --temperature 0.6
"""

import argparse

import torch

from llmbr.device import get_device


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
             top_k: int | None = 50, eot_id: int | None = None) -> torch.Tensor:
    """Autoregressive sampling: predict one token, append it, repeat.

    temperature  < 1 = safer/more repetitive, > 1 = more random
    top_k        sample only among the k most likely tokens (cuts off nonsense tail)
    eot_id       stop when the model ends the speech
    """
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = idx[:, -model.context_length:]          # model only sees the last T tokens
        logits = model(context)[:, -1, :]                 # logits for the next token only
        if temperature <= 0:                              # greedy decoding
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, -torch.inf)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        if eot_id is not None and (next_id == eot_id).all():
            break
    model.train(was_training)
    return idx


def main() -> None:
    from llmbr.model.gpt import GPTModel
    from llmbr.tokenizer import load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--prompt", default="Senhor Presidente,")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # The checkpoint carries its own config + meta, so it rebuilds the exact model.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg, meta = ckpt["cfg"], ckpt["meta"]
    device = get_device(cfg["train"]["device"])
    model = GPTModel(vocab_size=meta["vocab_size"], **cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.to(device)

    tok = load_tokenizer(cfg)
    ids = tok.encode(args.prompt, add_special_tokens=False).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k,
                   eot_id=meta["eot_id"])
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
