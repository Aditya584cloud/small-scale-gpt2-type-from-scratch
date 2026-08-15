import time
from tokenizer.bpe import BPETokenizer

with open("data/tinystories_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer.load(
    "data/tinystories_bpe_1024.json"
)

sample = text[:100_000]

sample_bytes = len(sample.encode("utf-8"))

tokenizer.encode(sample)

times = []

for _ in range(5):
    start = time.perf_counter()

    tokenizer.encode(sample)

    elapsed = time.perf_counter() - start
    times.append(elapsed)

average_time = sum(times) / len(times)
throughput = sample_bytes / average_time / 1e6
print("Sample bytes:", sample_bytes)
print("Times:", times)
print("Average time:", average_time)
print("Throughput:", throughput, "MB/s")