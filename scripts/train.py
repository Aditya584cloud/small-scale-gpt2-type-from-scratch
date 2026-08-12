import torch
from model.gpt import GPT, GPTConfig
from datasets.gpt_dataset import GPTDataset
from tokenizer.character import CharacterTokenizer
from torch.utils.data import DataLoader

text = "Hello World!"*100
tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

config = GPTConfig(
    vocab_size = tokenizer.vocab_size(),
    context_length = 16,
    embedding_dim = 128,
    num_heads = 4,
    num_layers = 4,
    dropout = 0.1
)

model = GPT(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

dataset = GPTDataset(
    tokens,
    context_length=config.context_length,
)

loader = DataLoader(
    dataset,
    batch_size = 4,
    shuffle = True
)

model.train()
for epoch in range(500):
    for  x, targets in loader:
        optimizer.zero_grad()

        logits, loss = model(x, targets)

        loss.backward()
        optimizer.step()


        if epoch % 50 == 0:
            print(loss.item())


        break