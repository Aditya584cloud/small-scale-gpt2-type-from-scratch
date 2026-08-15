import numpy as np

from tokenizer.bpe import BPETokenizer

with open("data/tinystories_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = BPETokenizer.load(
    "data/tinystories_bpe_1024.json"
)

split = int(0.9 * len(text))

train_text = text[:split]
val_text = text[split:]


print("Train characters:", len(train_text))
print("Val characters:", len(val_text))

print("Encoding train set...")
train_ids = tokenizer.encode(train_text)

print("Encoding validation set...")
val_ids = tokenizer.encode(val_text)



train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)



train_ids.tofile("data/tinystories_train.bin")
val_ids.tofile("data/tinystories_val.bin")


print("Train tokens:", len(train_ids))
print("Val tokens:", len(val_ids))

print("Train file size:", train_ids.nbytes / 1e6, "MB")
print("Val file size:", val_ids.nbytes / 1e6, "MB")