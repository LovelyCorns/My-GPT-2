import pyarrow.parquet as pq
import torch
from tokenizer import Tokenizer

path = './fineweb/004_00000.parquet'


class PreTrainData:

    def __init__(self, train_rate):
        self.tokenizer = Tokenizer()
        all = self.load_dataset()
        train_len = int(len(all) * train_rate)
        self.train = all[:train_len]
        self.valid = all[train_len:]

    def load_dataset(self):
        parquet_file = pq.ParquetFile(path)
        return [self.tokenizer.encode(parquet_file.read_row_group(i)[0][0].as_py()) for i in
                range(parquet_file.num_row_groups)]

    def get_batch(self, use_type='train', block_size=1024, seq_size=1):
        data = self.train if use_type == 'train' else self.valid

        data_ix = torch.randint(len(data), (1,))
        data = data[data_ix]
        ix = torch.randint(len(data) - block_size, (seq_size,))
        x = torch.stack([torch.tensor(data[i: i + block_size]) for i in ix])
        y = torch.stack([torch.tensor(data[i + 1: i + block_size + 1]) for i in ix])
        return x, y


if __name__ == '__main__':
    data = PreTrainData(0.9)
    x, y = data.get_batch("vaild", 64, 2)
    print(x.shape)
    print(y.shape)
