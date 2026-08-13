import torch
from model.gpt import GPT, GPTConfig
from tokenizer.character import CharacterTokenizer

text = "Hello World!" * 100
tokenizer = CharacterTokenizer(text)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size(),
    context_length=16,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.1
)

fresh_model = GPT(config)

state = torch.load("checkpoints/gpt_v0.pt")
fresh_model.load_state_dict(state)

fresh_model.eval()

print("Checkpoint loaded successfully")