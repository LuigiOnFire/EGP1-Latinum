import os
from torch.utils.data import Dataset

def TokenizedLatinDataset(Dataset):

    def __init__(self, token_data_dir, block_size):
        self.block_size = block_size
        self.bin_files = []
        self.data_len = None
        encode_suffix = "encoded.bin"
        for file in os.listdir(self.token_data_dir)
           if file.endswith(encode_suffix):
               self.bin_files.append(self.token_data_dir, file))


    def __len__(self):
        if self.data_len != None:
            return self.data_len
        else:
            tot_len = 0
            for bin_file in self.bin_files:
                with open(bin_file, 'r') as f:
                    token_data = np.memmap(bin_file), dtype=np.uint16, mode='r')
                    tot_len += len(token_data) - self.block_size

            return tot_len

    def __get_item__(self, idx):
        running_len = 0
        for bin_file in self.bin_files:
            with open(bin_file, 'r') as f:
                token_data = np.memmap(bin_file), dtype=np.uint16, mode='r')
                effective_len = len(token_data) - self.block_size
                if idx < running_len + effective_len:
                    data_start = idx - running_len 
                    x = token_data[data_start:data_start + self.block_size] 
                    y = token_data[data_start + self.block_size]
                    return x, y

                running_len += effective_len
