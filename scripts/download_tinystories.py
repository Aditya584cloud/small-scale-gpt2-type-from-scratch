from datasets import load_dataset

ds = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

with open("data/tinystories_sample.txt", "w", encoding="utf-8") as f:
    for example in ds.take(10000):
        f.write(example["text"])
        f.write("\n")

