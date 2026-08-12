import torch
import torch.nn as nn

from dataclasses import dataclass
@dataclass

class GPTConfig:
    vocab_size: int
    context_length: int = 16
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0 :
            raise ValueError("embedding_dim must be divisible by num_heads")


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.head_dim = self.embedding_dim // self.num_heads

        self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.context_length, config.context_length))
        )

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        

        wei = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out, wei
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.attention = AttentionHead(config)
        

        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            )
        
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        attention_out, _ = self.attention(self.ln1(x))
        x = x + self.dropout(attention_out)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.num_layers)]
            )

        self.ln_f = nn.LayerNorm(config.embedding_dim)

        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)


    def forward(self, x, targets = None):

        _, T = x.size()
        token_embeddings = self.token_embedding(x)

        position_ids = torch.arange(T, device=x.device)
        position_embeddings = self.position_embedding(position_ids)

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

        return logits, loss


config = GPTConfig(50)
model = GPT(config)

