from torch.utils.data import DataLoader
from dataset.gpt_dataset import GPTDataset

context_length = 128
batch_size = 32

train_dataset = GPTDataset(
    "data/tinystories_train.bin",
    context_length
)

val_dataset = GPTDataset(
    "data/tinystories_val.bin",
    context_length
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

x, y = next(iter(train_loader))

print(x.shape)
print(y.shape)
print(x.dtype)
print(y.dtype)

print(x[0])
print(y[0])