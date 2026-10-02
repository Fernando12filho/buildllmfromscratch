import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import TensorDataset, DataLoader

CONTEXT_LENGTH = 256   # tokens per chunk (can't exceed the rows of pos_emb)
EMB_DIM = 768          # numbers in each token's vector
VOCAB_SIZE = 50257     # size of the GPT-2 vocabulary
BATCH_SIZE = 4         # chunks handed out per batch

        
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
    dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


def load_token_ids(path):
    """Read the text file and turn it into a list of GPT-2 token IDs."""
    tokenizer = tiktoken.get_encoding("gpt2")   # loads GPT-2's ready-made BPE vocabulary (50,257 tokens)
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return tokenizer.encode(raw_text)           # str -> list of ints, e.g. [40, 367, 2885, ...]


def make_chunks(token_ids, max_length, stride):
    """Cut the ID list into equal chunks. targets = inputs shifted one token to the right."""
    inputs, targets = [], []
    for i in range(0, len(token_ids) - max_length, stride):
        inputs.append(token_ids[i : i + max_length])
        targets.append(token_ids[i + 1 : i + max_length + 1])
    return torch.tensor(inputs), torch.tensor(targets)   # both [num_chunks, max_length]


def simple_attention(h):
    """Self-attention without trainable weights.
    h: [batch, tokens, emb_dim]. Returns (context, weights).
    Each chunk in the batch is handled separately: tokens only look at tokens in their own chunk."""
    scores = h @ h.transpose(-2, -1)          # [B, T, T]  every token dotted with every token of its chunk
    weights = torch.softmax(scores, dim=-1)   # [B, T, T]  each row becomes percentages that sum to 1
    context = weights @ h                     # [B, T, D]  each token = weighted blend of its chunk's tokens
    return context, weights

if __name__ == "__main__":
    torch.manual_seed(123)   # same random tables and same shuffle order on every run

    token_ids = load_token_ids("the-veridict.txt")
    print(len(token_ids))
    inputs, targets = make_chunks(token_ids, max_length=CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    print("chunks:", inputs)                           # [20, 256]

    dataset = TensorDataset(inputs, targets)                 # pairs chunk i with its targets i
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  # 4 chunks at a time


    tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)              # token table    [50257, 768]
    pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM)          # position table [256, 768]
    
    d_in, d_out = EMB_DIM, 64
    torch.manual_seed(123)
    ca = CausalAttention(d_in, d_out, CONTEXT_LENGTH, 0.0)

    all_context_vecs = []
    for x, y in loader:                                      # 5 batches; x and y are [4, 256] token IDs
        tok_vec = tok_emb(x)                                 # [4, 256, 768]  row lookup
        pos_vec = pos_emb(torch.arange(x.shape[1]))          # [256, 768]     one row per slot
        inp_vec = tok_vec + pos_vec                          # [4, 256, 768]  positions reused for all 4 chunks
        context, weights = simple_attention(inp_vec)         # [4, 256, 768]
        all_context_vecs.append(context)                     # y isn't used yet: it's the answer key for the loss

    all_context_vecs = torch.cat(all_context_vecs)           # [20, 256, 768]  every token of every chunk
    print("context vectors:", all_context_vecs.shape)
    print("row sums of weights:", weights[0].sum(dim=-1)[:5])                    # all 1.0
    print("weight each token gives itself:", weights.diagonal(dim1=-2, dim2=-1).mean().item())
    
    
    
