import torch
import torch.nn as nn
import tiktoken
import numpy
from torch.utils.data import TensorDataset, DataLoader


tokenizer = tiktoken.get_encoding("gpt2") ## what this get encoding do? You choose a tokenizer encoding, which 
batch = []
with open("the-veridict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
token_ids = tokenizer.encode(raw_text)

def make_chunks(token_ids, max_lenght, stride):
    inputs, targets = [], []
    for i in range(0, len(token_ids) - max_lenght, stride):
        inputs.append(token_ids[i: i + max_lenght])
        targets.append(token_ids[i + 1: i + max_lenght + 1])
    return torch.tensor(inputs), torch.tensor(targets)

inputs, targets = make_chunks(token_ids, max_lenght=256, stride=256)

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

x, y = next(iter(loader))
print(x[0, :10])                          # 10 token IDs
print(tokenizer.decode(x[0, :10].tolist()))  # the text they represent
print(tokenizer.decode(y[0, :10].tolist()))

# GPT_CONFIG_124M ={
#     "vocab_size": 50257, 
#     "context_length": 1024,
#     "emb_dim": 768,
#     "n_heads": 12,
#     "n_layers": 12, 
#     "drop_rate": 0.1, 
#     "qvk_bias": False   #query-key_value bias
# }

# for i in range()
    
# # batch.append(torch.tensor(tokenizer.encode(raw_text)))
# # batch = torch.stack(batch, dim=0)
    
# print(batch)

# class DummyGPTModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
#         self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
#         self.drop_emb = nn.Dropout(cfg["drop_rate"])
#         self.trf_blocks = nn.Sequential(
#         *[DummyTransformerBlock(cfg)
#             for _ in range(cfg["n_layers"])]
#         )
#         self.final_norm = DummyLayerNorm(cfg["emb_dim"])
#         self.out_head = nn.Linear(
#             cfg["emb_dim"], cfg["vocab_size"], bias=False
#         )
#     def forward(self, in_idx):
#         batch_size, seq_len = in_idx.shape
#         tok_embeds = self.tok_emb(in_idx)
#         pos_embeds = self.pos_emb(
#             torch.arange(seq_len, device=in_idx.device)
#         )
#         x = tok_embeds + pos_embeds
#         x = self.drop_emb(x)
#         x = self.trf_blocks(x)
#         x = self.final_norm(x)
#         logits = self.out_head(x)
#         return logits
        
# class DummyTransformerBlock(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
        
#     def forward(self, x):
#         return x

# class DummyLayerNorm(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-5):
#         super().__init__()
        
#     def forward(self, x):
#         return x
    

# # torch.manual_seed(123)
# # model = DummyGPTModel(GPT_CONFIG_124M)
# # logits = model(batch)
# # print("output Shape: ", logits.shape)
# # print(logits)

        