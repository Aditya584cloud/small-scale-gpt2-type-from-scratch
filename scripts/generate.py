import torch
import torch.nn as nn
from model.gpt import GPTConfig, GPT
from tokenizer.bpe import BPETokenizer

tokenizer = BPETokenizer.load("data/tinystories_bpe_1024.json")

config = GPTConfig(
    vocab_size=1024,
    context_length=128,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.1,
)

fresh_gpt = GPT(config)

state = torch.load("checkpoints/gpt_bpe_1024.pt", map_location = "cpu")
fresh_gpt.load_state_dict(state)

fresh_gpt.eval()

prompt = "Once upon a time"

token_ids = torch.tensor(tokenizer.encode(prompt))

token_ids = token_ids.view(1,-1)
with torch.no_grad():
    for i in range(100):
        if token_ids.size(1) > config.context_length:
            token_ids = token_ids[:, -config.context_length:]

        logits, _ = fresh_gpt(token_ids)

        new_logits = logits[:, -1, :]

        probs = torch.nn.functional.softmax(new_logits, dim = -1)

        next_token = torch.argmax(probs, dim = -1)
        
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim = 1)
        
    generated_ids = token_ids.squeeze(0).tolist()
    text = tokenizer.decode(generated_ids)
    print(text)
