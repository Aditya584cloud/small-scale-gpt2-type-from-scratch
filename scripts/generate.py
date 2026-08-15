import torch
import torch.nn as nn
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

prompt = "Hello"

token_ids = torch.tensor(tokenizer.encode(prompt))

token_ids = token_ids.view(1,-1)
with torch.no_grad():
    for i in range(20):
        if token_ids.size(1) > config.context_length:
            token_ids = token_ids[:, -config.context_length:]

        logits, _ = fresh_gpt(token_ids)

        new_logits = logits[:, -1, :]

        probs = torch.nn.functional.softmax(new_logits, dim = -1)

        next_token = torch.argmax(probs, dim = -1)
        
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim = 1)
        
    generated_ids = token_ids.squeeze(0).tolist()
    char = tokenizer.decode(generated_ids)
    print(char)
