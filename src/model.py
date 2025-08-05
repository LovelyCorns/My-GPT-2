import torch
import torch.nn as nn

from src.data import PreTrainData
from tokenizer import Tokenizer


class Model(nn.Module):

    def __init__(self, n_ctx, max_token, n_embd, n_head, n_layer, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.tokenizer = Tokenizer()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.max_token = max_token
        self.wte = nn.Embedding(self.tokenizer.vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)

    def forward(self, ids, ):
        B, T = ids.shape
        tok_emb = self.wte(ids)
        pos_emb = self.wpe(torch.arange(T, device=self.device))
        print(tok_emb.shape)
        print(pos_emb.shape)
        x = tok_emb + pos_emb
        print(x.shape)

    # def attn(self, ):


if __name__ == '__main__':
    model = Model(32, 5, 16, 1, 1, device='cpu')
    data = PreTrainData(0.9)
    X, Y = data.get_batch(block_size=32)
    model.forward(X)
