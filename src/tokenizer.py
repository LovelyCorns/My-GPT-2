import tiktoken as tk

encoding_name = "o200k_base"


class Tokenizer:

    def __init__(self):
        self.enc = tk.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, txt: str):
        return self.enc.encode(txt)

    def decode(self, tks):
        return self.enc.decode(tks)


