import os
import random
import torch

import numpy as np
from torch.utils.data import Dataset

from utils import PAD_INDEX

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
                token_data = np.memmap(bin_file, dtype=np.uint16, mode='r')
                tot_len += len(token_data) - self.block_size
            self.data_len = tot_len
            return tot_len

    def __getitem__(self, idx):
        running_len = 0 
        if len(self.bin_files) == 1:
            bin_file = self.bin_files[0]
            token_data = np.memmap(bin_file, dtype=np.uint16, mode='r')
            x = torch.from_numpy(np.array(token_data[idx:idx + self.block_size]).astype(np.int64))
            y = torch.from_numpy(np.array(token_data[idx + 1:idx + self.block_size + 1]).astype(np.int64))
            # x, y = self._pad_item(x, y) Removing pads for now, possibly doing more harm than good
            return x, y

        else:   
            for bin_file in self.bin_files:
                token_data = np.memmap(bin_file, dtype=np.uint16, mode='r')
                effective_len = len(token_data) - self.block_size
                if idx < running_len + effective_len:
                    data_start = idx - running_len
                    x = np.copy(token_data[data_start:data_start + self.block_size])
                    y = np.copy(token_data[data_start + 1:data_start + self.block_size + 1])
                    x = torch.from_numpy(x)
                    y = torch.from_numpy(y)
                    # x, y = self._pad_item(x, y) Removing pads for now, possibly doing more harm than good
                    return x, y

                running_len += effective_len

    def _pad_item(self, x, y):
        xn = x.clone()
        yn = y.clone()
        max_pads = self.block_size / 8 - 1

        num_pads = random.randint(0, max_pads) 

        # padding out everything would be pretty bad
        assert num_pads < self.block_size - 1, \
            "num_pads value set too high and will pad out whole sample"

        # x = [X like fluffy]
        # y = [like fluffy dogs]
        for idx in range(num_pads): # max value: self.block_size - 3
            xn[idx] = PAD_INDEX
            yn[idx] = PAD_INDEX

        xn[num_pads] = PAD_INDEX # max value: self.block_size - 2
        #print(xn)
        #print(yn)

        assert xn[-1] != PAD_INDEX, "WARNING: Input vector is all pads!"
        assert yn[-1] != PAD_INDEX, "WARNING: Output vector is all pads!"
        return xn, yn
