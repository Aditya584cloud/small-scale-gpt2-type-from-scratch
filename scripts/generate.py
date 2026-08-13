import torch
from model.gpt import GPTConfig, GPT
from tokenizer.character import CharacterTokenizer

text = "Hello World!"*100
tokenizer = CharacterTokenizer(text)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size(),
    context_length=16,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.1
)

fresh_gpt = GPT(config)

state = torch.load("checkpoints/gpt_v0.pt")
fresh_gpt.load_state_dict(state)

fresh_gpt.eval()