import torch
from model.gpt import GPT, GPTConfig
from tokenizer.character import CharacterTokenizer

text = "Hello World!" * 100
tokenizer = CharacterTokenizer(text)

device = 'cuda' if torch.cuda.is_available() else cpu
checkpoint = torch.load("checkpoints/gpt_v2.pt", map_location = device)



config = GPTConfig(**checkpoint["config"])

fresh_model = GPT(config)

fresh_model.load_state_dict(checkpoint["model_state"])
fresh_model.eval()

print("Checkpoint loaded successfully")