import tiktoken
import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

tokenizer = tiktoken.get_encoding('gpt2')

with open("the-veridict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
token_ids = tokenizer.encode(raw_text)

def make_chunks(token_ids, max_length, stride):
    inputs, targets = [],[]
    for i in range(0, len(token_ids) - max_length, stride):
        inputs.append(token_ids[i: i + max_length])
        targets.append(token_ids[i + 1: i + max_length + 1])
    return torch.tensor(inputs), torch.tensor(targets)

def __init__():
    inputs, targets = make_chunks(token_ids, 256, 256)
    tok_emb = nn.Embedding(vocab_size, embedding_dim)
    pos_emb = nn.Embedding(ctx_lenght, emb=dim)
    
    dataset = TensorDataset(inputs, targets) ## matches one with the other
    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True) ## loads batches from tensor dataset
    
    for i in range()
    
    
    
    



