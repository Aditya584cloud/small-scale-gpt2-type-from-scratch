import torch

from dataset.gpt_data import GPTData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = GPTData("data/tinystories_train.bin")

x, y = train_data.get_batch(batch_size=32, context_length=128, device=device)

print("device:", device)
print("x shape:", x.shape)
print("y shape:", y.shape)
print("x dtype:", x.dtype)
print("y dtype:", y.dtype)

print("x[0]:", x[0])
print("y[0]:", y[0])

print(
    "shift correct:",
    torch.equal(x[:, 1:], y[:, :-1])
)