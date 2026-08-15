import time

from tokenizer.bpe import BPETokenizer


with open("data/tinystories_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = BPETokenizer.load("data/tinystories_bpe_1024.json")

sample = text[:100_000]
original_bytes = len(sample.encode("utf-8"))

start = time.time()

encoded = tokenizer.encode(sample)

elapsed = time.time() - start

print("Original bytes:", original_bytes)
print("BPE tokens:", len(encoded))
print("Bytes per token:", original_bytes / len(encoded))
print("Encoding time:", elapsed, "seconds")
print("Encoding speed:", original_bytes / elapsed / 1e6, "MB/s")

