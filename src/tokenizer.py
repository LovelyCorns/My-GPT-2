import tiktoken as tk

class Tokenizer:

    def __init__(self, encoding_name="gpt2"):
        self.enc = tk.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, txt: str):
        return self.enc.encode(txt, allowed_special={"<|endoftext|>"})

    def decode(self, tks):
        return self.enc.decode(tks)


