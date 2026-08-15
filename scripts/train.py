import torch
from model.gpt import GPT, GPTConfig
from datasets.gpt_dataset import GPTDataset
from tokenizer.character import CharacterTokenizer
from torch.utils.data import DataLoader

text = "Hello World!"*100
tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

n = int(0.9 * (len(tokens)))
train_data = tokens[:n]
val_data = tokens[n:]

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

train_dataset = GPTDataset(
    train_data,
    context_length=config.context_length,
)

val_dataset = GPTDataset(
    val_data,
    context_length=config.context_length,
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 4,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 4,
    shuffle = True
)

def evaluate(model, val_loader):
    model.eval()
    total_loss_val = 0
    with torch.no_grad():
        for x, targets in val_loader:
            
            logits, loss_val = model(x, targets)

            total_loss_val += loss_val.item()
        
        av_loss_val = total_loss_val / len(val_loader)
        

    return av_loss_val

def train(model, optimizer, train_loader):
    model.train()
    total_loss_train = 0
    for  x, targets in train_loader:
        
        optimizer.zero_grad()

        logits, loss_train = model(x, targets)

        total_loss_train += loss_train.item()
        loss_train.backward()
        optimizer.step()

    

    av_loss_train = total_loss_train / len(train_loader)
    
    return av_loss_train


for epoch in range(100):
    
    train_loss = train(model,optimizer, train_loader)
    val_loss = evaluate(model, val_loader)
   
    
    if epoch % 50 == 0:
        print(f"step {epoch}: train loss {train_loss}, val loss {val_loss}")



checkpoint = {
    "model_state": model.state_dict(),
    "config": {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "embedding_dim": config.embedding_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    },
}

torch.save(checkpoint, "checkpoints/gpt_v2.pt")