"""TEMPLATE — training loop.

Run:  python -m llmbr.train --config configs/small.toml [--resume]

Steps to implement:
  1. cfg = load_config(args.config); device = get_device(cfg["train"]["device"])
     torch.manual_seed(cfg["train"]["seed"])
  2. Data: np.memmap("data/processed/train.bin", dtype=np.uint16, mode="r")
     get_batch(split): pick batch_size random offsets, x = data[i:i+T], y = data[i+1:i+T+1]
  3. Model: GPTModel(vocab_size=tokenizer vocab, **cfg["model"]).to(device)
  4. Optimizer: AdamW(lr, weight_decay); LR schedule = linear warmup then cosine decay.
  5. Loop for max_steps:
       - loss = cross_entropy(logits.flatten(0, 1), y.flatten())
       - loss / grad_accum, backward; every grad_accum steps: clip_grad_norm_(1.0), step
       - on CUDA use torch.autocast(device_type="cuda", dtype=torch.bfloat16) for ~2x speed
       - every eval_interval: average val loss over eval_iters batches, print it,
         and generate a short sample from "Senhor Presidente," to watch progress
       - every save_interval: torch.save({model, optimizer, step, cfg}, checkpoints/last.pt)
  6. --resume: load checkpoints/last.pt and continue from its step.

Compare with your ../train.py — same idea, but streaming from .bin and checkpointing.
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.toml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    # TODO: steps 1–6 above.
    raise NotImplementedError


if __name__ == "__main__":
    main()
