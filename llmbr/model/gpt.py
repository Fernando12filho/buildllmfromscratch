"""GPT model (decoder-only transformer), sized by cfg["model"].

    token ids (B, T)
      -> tok_emb + pos_emb                    (B, T, emb_dim)
      -> dropout
      -> n_layers × TransformerBlock          (B, T, emb_dim)
      -> final LayerNorm
      -> out_head (weights tied to tok_emb)   (B, T, vocab_size)  = logits

Usage:
    model = GPTModel(vocab_size=meta["vocab_size"], **cfg["model"])
    logits = model(idx)          # idx: (B, T) int64

Parameter count for small.toml (vocab 32k, emb 384, 6 layers) ≈ 23M with tied
weights: 6 blocks × 12·384² ≈ 10.6M + tok_emb 32000×384 ≈ 12.3M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal self-attention with n_heads heads.

    Same math as the single-head CausalAttention in study/gpt2.py:
        softmax(Q·Kᵀ / sqrt(d_head) + causal mask) · V
    but (1) Q, K, V come from one fused Linear, (2) emb_dim is split into n_heads
    smaller heads that attend independently, and (3) the attention itself runs in
    F.scaled_dot_product_attention, which applies the causal mask (is_causal=True)
    and uses fast fused kernels on CUDA and MPS — no mask buffer needed.
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float, qkv_bias: bool):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # mixes the heads back together
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                          # 3 × (B, T, C)
        # (B, T, C) -> (B, n_heads, T, head_dim): each head attends separately
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)          # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)           # concat heads
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Per-token MLP: expand ×4, GELU, project back."""

    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block with residual ("shortcut") connections:
        x = x + attn(norm1(x))
        x = x + ff(norm2(x))
    The residual path lets gradients flow straight through deep stacks.
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float, qkv_bias: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, n_heads, dropout, qkv_bias)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, emb_dim: int,
                 n_heads: int, n_layers: int, dropout: float = 0.1,
                 qkv_bias: bool = False):
        super().__init__()
        self.context_length = context_length
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)       # token id  -> vector
        self.pos_emb = nn.Embedding(context_length, emb_dim)   # position  -> vector
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(emb_dim, n_heads, dropout, qkv_bias) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
        # Weight tying: the output layer reuses the token-embedding matrix.
        # Saves vocab_size × emb_dim params (12M here) and usually trains better on small data.
        self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the init of layers that write into the residual stream,
        # so the sum over n_layers blocks doesn't blow up at the start of training.
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "net.2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.context_length:
            raise ValueError(f"sequence length {T} > context_length {self.context_length}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)                                 # (B, T, vocab_size)

    def num_params(self) -> int:
        # Tied weights are one tensor, so parameters() already counts them once.
        return sum(p.numel() for p in self.parameters())
