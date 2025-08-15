import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data import PreTrainData
from tokenizer import Tokenizer


@dataclass
class ModelConfig:
    n_embd: int = 32
    n_head: int = 2
    n_ctx: int = 1024
    n_layer: int = 1
    device: str = "cpu"

class Model(nn.Module):

    def __init__(self, n_ctx, max_token, n_embd, n_hc, p, n_layer, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.tokenizer = Tokenizer()
        self.n_layer = n_layer
        self.n_hc = n_hc
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.max_token = max_token
        self.wte = nn.Embedding(self.tokenizer.vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.blocks = Block(n_ctx, n_embd, n_hc, p)

    def forward(self, ids):
        T, C = ids.shape
        # B, T, C = ids.shape
        tok_emb = self.wte(ids)
        pos_emb = self.wpe(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        print(x[::-1])


class Block(nn.Module):

    def __init__(self, n_ctx, n_embd, n_hc, p: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MA(n_hc, n_ctx, n_embd, p)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MA(nn.Module):

    def __init__(self, n_hc, n_ctx, n_embd, p: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_embd % n_hc != 0:
            raise Exception("n_embd can not be divide by n_hc")
        self.n_hc = n_hc
        self.n_embd = n_embd
        self.heads = nn.ModuleList([Head(n_ctx, n_embd, int(n_embd / n_hc)) for _ in range(n_hc)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        r = torch.cat([self.heads[i](x) for i in range(self.n_hc)], dim=-1)
        return self.proj(r)


class Head(nn.Module):

    def __init__(self, n_ctx, n_embd, n_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = nn.Linear(n_embd, n_dim)
        self.query = nn.Linear(n_embd, n_dim)
        self.value = nn.Linear(n_embd, n_dim)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)  # (B, T, D)
        q = self.query(x)  # (B, T, D)
        v = self.value(x)  # (B, T, D)
        k = k.transpose(1, 2)  # (B, D, T)
        k_q = q @ k  # (B, T, T)
        k_q = k_q * (1 / math.sqrt(k.shape[1]))
        k_q = k_q.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        k_q = torch.softmax(k_q, -1)
        k_q = self.dropout(k_q)
        r = k_q @ v
        print(f"结果形状: {r.shape}")
        return r


if __name__ == '__main__':
    config = ModelConfig()
    model = Model(config.n_ctx, 5, config.n_embd, config.n_head, config.n_layer, config.device)
    data = PreTrainData(0.9)
    X, Y = data.get_batch(block_size=32)
    X = X.to(device=config.device)
    Y = Y.to(device=config.device)
    print(X.shape)
    model.forward(X)
    # head = Head(32, 8, 16)

    # hs = MA(n_hc=12, n_ctx=32, n_embd=768)
    # hs(torch.randn(1, 32, 768))
    # head.forward(torch.randn(1, 32, 8))
