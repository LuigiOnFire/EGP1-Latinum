import os
from torch.utils.data import Dataset

def TokenizedLatinDataset(Dataset):

    def __init__(self):
        self.token_data_dir = Path(__file__).parent.parent / "data" / "token_data"
        self.bin_files = []
        encode_suffix = "encoded.bin"
        for file in os.listdir(self.token_data_dir)
           if file.endswith(encode_suffix):
               self.bin_files.append(self.token_data_dir, file))


    def __len__(self):
        return len(self.bin_files)

    def __get_item__(self, idx):
        bin_file = self.bin_files[idx]
        with open(bin_file, 'r') as f:
            token_data = np.memmap(bin_file), dtype=np.uint16, mode-'r')

def BatchSampler():
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        current_bin_file = None
        current_indices = 
