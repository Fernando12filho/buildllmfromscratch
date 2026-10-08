import urllib.request
import re

with open("the-veridict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of character: ", len(raw_text))
# print(raw_text[:99])

# print("==========================================================================================================")
# token1 = re.split(r'([,.]|\s)', raw_text)
# print(token1[:100])
# token1 = [item for item in token1 if item.strip()]
# print("New list: ")
# print(token1[:100])

# print("==========================================================================================================")
# token2 = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# token2 = [item.strip() for item in token2 if item.strip()]
# print(token2[:100])

print("==========================================================================================================")
prepocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
prepocessed = [item.strip() for item in prepocessed if item.strip()] ## how does this linde of code works?
print(len(prepocessed))
print(prepocessed[:30])

## list of all unique tokens and sort them alphatically to determince the vocabulary size
all_words = sorted(set(prepocessed)) ## collapses duplicates, sort turn that into alphatically sorted list
vocab_size = len(all_words)
print(all_words[:99])
print(vocab_size)


print("==========================================================================================================")
vocab = {token:integer for integer, token in enumerate(all_words)} ## At what point did we eliminated the duplicates?
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50 :
        break 
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

print("==========================================================================================================")
tokenizer = SimpleTokenizerV1(vocab)
text = "In the sunlit terraces of the"
ids = tokenizer.encode(text)
print(ids)
print("==========================================================================================================")
back = tokenizer.decode(ids)
print(back)


## Include special tokens, and start to add some context <|endoftext|>
all_tokens = sorted(list(set(prepocessed)))
all_tokens.extend({"<|endoftext|>", "<|unk|>"})
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items()) [-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

text1 = "hello, do you like tea?"
text2 = "In te sunlit terraces of the palace."
text = "<|endoftext|>".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))








