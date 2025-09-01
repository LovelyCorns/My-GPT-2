import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import PreTrainData
from tokenizer import Tokenizer


@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_ctx: int = 1024
    n_layer: int = 12
    device: str = "cuda"
    interval: int = 100
    max_iter: int = 50 * interval
    lr = 2.5e-4
    eval_iter: int = interval
    p: float = 0.1
    dataset_path = './fineweb/000_00007.parquet'
    model_checkpoints_path = './checkpoints/gpt2_base_20250825_104521' # it could not be null if continue_pretrain is true
    continue_pretrain = True
    patience = 3


class Model(nn.Module):

    def __init__(self, n_ctx, max_token, n_embd, n_hc, p, n_layer, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.n_ctx = n_ctx
        self.tokenizer = Tokenizer()
        self.max_token = max_token
        self.wte = nn.Embedding(self.tokenizer.vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.blocks = nn.Sequential(*[Block(n_ctx, n_embd, n_hc, p) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.tokenizer.vocab_size)

    def forward(self, ids, labels=None):
        B, T = ids.shape
        tok_emb = self.wte(ids)
        pos_emb = self.wpe(torch.arange(T, device=self.device))
        x = self.ln(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)

        if labels is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)
            loss = F.cross_entropy(logits, labels)
            return logits, loss
        else:
            return logits

    def gen(self, ids):
        ids = ids.to(self.device)

        for i in range(self.max_token):
            logits = self(ids)
            last_embd = logits[:, -1:]
            last_embd = F.softmax(last_embd.squeeze(1), dim=-1)
            sample = torch.multinomial(last_embd, num_samples=1)
            ids = torch.cat((ids, sample), dim=1)

            # only do tailor last 1k context when 2nd dim overflow
            if ids.shape[1] > self.n_ctx:
                ids = ids[:, -self.n_ctx:]

        return ids


class Block(nn.Module):

    def __init__(self, n_ctx, n_embd, n_hc, p: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MA(n_hc, n_ctx, n_embd, p)
        self.mlp = MLP(n_embd, p)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class MLP(nn.Module):

    def __init__(self, n_embd, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p)
        )

    def forward(self, x):
        return x + self.mlp(x)


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
        return r


#  test
if __name__ == '__main__':
    config = ModelConfig()
    model = Model(config.n_ctx, 5, config.n_embd, config.n_head, 0.1, n_layer=config.n_layer, device=config.device)
    data = PreTrainData(0.9)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    # for i in range(config.max_iter):
    #     if i % config.interval == 0:
    #         out = l_e(model, data, config)
    #         print(f"iter: {i}, out:{out}")
    #
    #     X, Y = data.get_batch(seq_size=32, batch_size=2)
    #     X = X.to(device=config.device)
    #     Y = Y.to(device=config.device)
    #     adamW = torch.optim.AdamW(model.parameters())
    #     # ids = model.gen(X)
    #     logits, loss = model(X, Y)
    #     adamW.zero_grad(set_to_none=True)
    #     loss.backward()
    #     adamW.step()

    # for b in range(ids.shape[0]):
    #     seq = ids[b, :]
    #     print(seq)
    #     print(model.tokenizer.decode(seq.tolist()))
    # model.tokenizer.decode()
    # head = Head(32, 8, 16)

    # hs = MA(n_hc=12, n_ctx=32, n_embd=768)
    # hs(torch.randn(1, 32, 768))
    # head.forward(torch.randn(1, 32, 8))
