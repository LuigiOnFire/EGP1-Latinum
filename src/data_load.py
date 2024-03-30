import numpy as np
import os
import torch

from torch.utils.data import Dataset

class TokenizedLatinDataset(Dataset):
    def __init__(self, token_data_dir, block_size):
        self.token_data_dir = token_data_dir
        self.block_size = block_size
        self.bin_files = []
        self.data_len = None
        encode_suffix = "encoded.bin"
        for file in os.listdir(self.token_data_dir):
            if file.endswith(encode_suffix):
                self.bin_files.append(self.token_data_dir / file)


    def __len__(self):
        if self.data_len != None:
            return self.data_len
        else:
            tot_len = 0
            for bin_file in self.bin_files:
                with open(bin_file, 'r') as f:
                    token_data = np.memmap(bin_file, dtype=np.int64)
                    tot_len += len(token_data) - self.block_size

            return tot_len

    def __getitem__(self, idx):
        running_len = 0
        for bin_file in self.bin_files:
            with open(bin_file, 'r') as f:
                token_data = np.memmap(bin_file, dtype=np.int64)
                effective_len = len(token_data) - self.block_size
                if idx < running_len + effective_len:
                    data_start = idx - running_len
                    x = torch.from_numpy(token_data[data_start:data_start + self.block_size])
                    y = torch.from_numpy(token_data[data_start + 1:data_start + self.block_size + 1])
                    return x, y

                running_len += effective_len
