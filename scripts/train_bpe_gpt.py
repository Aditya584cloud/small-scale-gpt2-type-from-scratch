import torch
from torch.utils.data import DataLoader

from dataset.gpt_data import GPTData
from model.gpt import GPTConfig, GPT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

context_length = 128
batch_size = 32

config = GPTConfig(
    vocab_size=1024,
    context_length=128,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.1,
)


train_data = GPTData("data/tinystories_train.bin")

val_data = GPTData("data/tinystories_val.bin")

model = GPT(config).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Parameters:", num_params)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 20
steps = 500
for epoch in range(epochs):

    model.train()
    train_loss = 0.0
    for step in range(steps):
        x, targets = train_data.get_batch(batch_size, context_length, device)

        optimizer.zero_grad()
        logits, loss = model(x, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if step % 50 == 0:
            print(
                f"epoch {epoch + 1} | "
                f"step {step}/{steps} | "
                f"loss {loss.item():.4f}"
            )
    train_loss /= steps

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _ in range(50):
            x, targets = val_data.get_batch(batch_size, context_length, device)
            logits, loss = model(x, targets)
            val_loss += loss.item()
    val_loss /= 50

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"train loss: {train_loss:.4f} | "
        f"val loss: {val_loss:.4f}"
    )

torch.save(model.state_dict(), "checkpoints/gpt_bpe_1024.pt")

print("Model saved.")