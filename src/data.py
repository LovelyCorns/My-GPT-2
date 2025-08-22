import pyarrow.parquet as pq
import torch

from tokenizer import Tokenizer


class PreTrainData:

    def __init__(self, train_rate, path):
        self.tokenizer = Tokenizer()
        self.path = path
        all = self.load_dataset()

        def concatenate_docs(docs):
            result = []
            for doc in docs:
                result.extend(doc)
                result.append(self.tokenizer.encode("<|endoftext|>")[0])
            return result

        all_tokens = concatenate_docs(all)
        train_len = int(len(all_tokens) * train_rate)

        self.train = all_tokens[:train_len]
        self.valid = all_tokens[train_len:]

    def load_dataset(self):
        parquet_file = pq.ParquetFile(self.path)
        return [self.tokenizer.encode(parquet_file.read_row_group(i)[0][0].as_py()) for i in
                range(parquet_file.num_row_groups)]

    def get_batch(self, use_type='train', seq_size=1024, batch_size=1):
        data = self.train if use_type == 'train' else self.valid

        if len(data) <= seq_size:
            raise ValueError(f"{use_type} dataset tokens which length is ({len(data)}) "
                             f"aren't able to generate each seq which size is {seq_size}")

        ix = torch.randint(len(data) - seq_size, (batch_size,))
        x = torch.stack([torch.tensor(data[i: i + seq_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i + 1: i + seq_size + 1]) for i in ix])
        return x, y


if __name__ == '__main__':
    data = PreTrainData(0.9)
    x, y = data.get_batch("vaild", 64, 2)
    print(x)
    print(y)
    print(x.shape)
    print(y.shape)
