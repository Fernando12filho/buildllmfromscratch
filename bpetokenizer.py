import tiktoken
import datasetv1
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-veridict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

def bpeexample():
    enc_sample = enc_text[800:]
    context_size = 100
    x = enc_sample[:context_size] # first 100 elements
    y = enc_sample[1:context_size+1] # Elementes 1 through 101
    print(f"x: {x}")
    print(f"y: {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "----> ", desired)

    for i in range(1, context_size):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "----> ", tokenizer.decode([desired]))

def create_dataloader_v1(txt, batch_size=4, max_lenght=256, stride=128, 
                            shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = datasetv1.GPTDatasetV1(txt, tokenizer, max_lenght, stride)
    dataloader = DataLoader (
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


input_ids = torch.tensor([2,3,5,1])
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_lenght=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape: \n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape) # The 8 × 4 × 256–dimensional tensor output shows that each token ID is now embed-ded as a 256-dimensional vector






    