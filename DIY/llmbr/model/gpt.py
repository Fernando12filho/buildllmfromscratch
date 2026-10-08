"""TEMPLATE — GPT model. Port your ../gpt2.py here.

Build it from the same blocks you already wrote, reading sizes from
cfg["model"] (configs/small.toml) instead of hard-coded GPT-2 numbers:

  MultiHeadAttention   causal self-attention (see ../selfattention.py)
                       tip: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, is_causal=True) is much faster on CUDA/MPS
  FeedForward          Linear(emb, 4*emb) -> GELU -> Linear(4*emb, emb)
  TransformerBlock     pre-LayerNorm: x = x + attn(ln1(x)); x = x + ff(ln2(x))
  GPTModel             token emb + position emb -> N blocks -> final LayerNorm -> out head

Parameter count for small.toml (vocab 16k, emb 384, 6 layers) ≈ 17M + 2×6M
embeddings ≈ 30M. Tie out_head.weight = tok_emb.weight to save ~6M.
"""

import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, emb_dim: int,
                 n_heads: int, n_layers: int, dropout: float, qkv_bias: bool = False):
        super().__init__()
        # TODO: tok_emb, pos_emb, drop, blocks (nn.Sequential of TransformerBlock),
        #       final_norm, out_head
        raise NotImplementedError

    def forward(self, idx):
        # idx: (batch, seq_len) token ids -> returns logits (batch, seq_len, vocab_size)
        raise NotImplementedError
