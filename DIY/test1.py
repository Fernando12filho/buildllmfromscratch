import tiktoken
import torch 

tokenizer = tiktoken.get_encoding('gpt2')

with open("../the-veridict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text = tokenizer.encode(raw_text) ## 
## print(enc_text)

