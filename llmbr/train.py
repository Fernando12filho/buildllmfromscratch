"""Training loop.

Run:  python -m llmbr.train --config configs/tiny.toml            # laptop MVP
      python -m llmbr.train --config configs/small.toml           # desktop
      python -m llmbr.train --config configs/tiny.toml --resume   # continue last.pt
      python -m llmbr.train --config configs/tiny.toml --max-steps 50   # smoke test

Saves to cfg["paths"]["checkpoints"]:
  last.pt   every save_interval steps (for --resume)
  best.pt   whenever validation loss improves (use this one for generation)
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from llmbr.config import load_config
from llmbr.data.loader import create_dataloader
from llmbr.device import get_device
from llmbr.generate import generate
from llmbr.model.gpt import GPTModel
from llmbr.tokenizer import load_tokenizer

SAMPLE_PROMPT = "Senhor Presidente,"


def lr_at(step: int, tc: dict) -> float:
    """Linear warmup to lr, then cosine decay down to 10% of lr at max_steps."""
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    progress = min(1.0, (step - tc["warmup_steps"]) /
                   max(1, tc["max_steps"] - tc["warmup_steps"]))
    return tc["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model, tc: dict, device) -> torch.optim.Optimizer:
    # Weight decay only on matrices (Linear/Embedding weights), not on biases or
    # LayerNorm parameters — decaying those just hurts.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"))


def loss_fn(model, x, y, autocast_ctx) -> torch.Tensor:
    with autocast_ctx():
        logits = model(x)
    # (B, T, V) vs (B, T) -> flatten to (B·T, V) vs (B·T,)
    return F.cross_entropy(logits.float().flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(model, loader, n_batches: int, device, autocast_ctx) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        losses.append(loss_fn(model, x.to(device), y.to(device), autocast_ctx).item())
    model.train()
    return sum(losses) / len(losses)


def infinite(loader):
    """Cycle over the DataLoader forever; each pass is one epoch (reshuffled)."""
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/small.toml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="override config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    tc, mc, paths = cfg["train"], cfg["model"], cfg["paths"]
    torch.manual_seed(tc["seed"])
    device = get_device(tc["device"])

    # bfloat16 autocast on CUDA roughly doubles speed and halves activation memory.
    # Needs an RTX 30xx or newer (Ampere+); older cards and MPS/CPU stay in float32.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        from contextlib import nullcontext as autocast_ctx

    processed = Path(paths["processed_dir"])
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    T, B = mc["context_length"], tc["batch_size"]
    train_dl = create_dataloader(str(processed / "train.bin"), B, T, shuffle=True)
    val_dl = create_dataloader(str(processed / "val.bin"), B, T,
                               shuffle=False, drop_last=False)
    if len(train_dl) == 0 or len(val_dl) == 0:
        raise SystemExit("not enough tokens for one batch — lower batch_size/context_length")

    model = GPTModel(vocab_size=meta["vocab_size"], **mc).to(device)
    opt = make_optimizer(model, tc, device)
    tok = load_tokenizer(cfg)

    ckpt_dir = Path(paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resumed from step {step} (best val {best_val:.3f})")

    def save(name: str) -> None:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best_val": best_val, "cfg": cfg, "meta": meta},
                   ckpt_dir / name)

    tokens_per_step = B * T * tc["grad_accum"]
    print(f"device: {device} | params: {model.num_params() / 1e6:.1f}M | "
          f"train tokens: {meta['train_tokens']:,} | {len(train_dl)} batches/epoch | "
          f"{tokens_per_step:,} tokens/step")

    batches = infinite(train_dl)
    model.train()
    t0 = time.time()
    while step < tc["max_steps"]:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tc)

        # Gradient accumulation: several small batches act like one big batch.
        for _ in range(tc["grad_accum"]):
            epoch, (x, y) = next(batches)
            loss = loss_fn(model, x.to(device), y.to(device), autocast_ctx)
            (loss / tc["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # tames loss spikes
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % 10 == 0:
            dt = time.time() - t0
            print(f"step {step:>6} | epoch {epoch} | loss {loss.item():.3f} | "
                  f"lr {lr_at(step, tc):.2e} | {10 * tokens_per_step / dt:,.0f} tok/s",
                  flush=True)
            t0 = time.time()

        if step % tc["eval_interval"] == 0 or step == tc["max_steps"]:
            val = evaluate(model, val_dl, tc["eval_iters"], device, autocast_ctx)
            # Perplexity = e^loss ≈ "how many tokens the model is choosing between".
            print(f"--- step {step} | val loss {val:.3f} | val ppl {math.exp(val):,.1f}")
            if val < best_val:
                best_val = val
                save("best.pt")
                print("    saved best.pt")
            prompt = torch.tensor([tok.encode(SAMPLE_PROMPT, add_special_tokens=False).ids],
                                  device=device)
            sample = generate(model, prompt, 60, temperature=0.8, top_k=50,
                              eot_id=meta["eot_id"])
            print("    sample:", tok.decode(sample[0].tolist()).replace("\n", " "))
            t0 = time.time()  # don't count eval time in tok/s

        if step % tc["save_interval"] == 0 or step == tc["max_steps"]:
            save("last.pt")

    print(f"done. best val loss {best_val:.3f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
