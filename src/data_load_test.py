from pathlib import Path
import torch

import data_load
import tokenizer_spm

config_train = {}
config_model = {}

config_train["token_data_dir"] = Path.home() / "latin-literature-dataset-170M" / "bin_train"
config_train["batch_size"] = 1
config_model["block_length"] = 16 # reduced from minigpt 


dataset = data_load.TokenizedLatinDataset(
        config_train["token_data_dir"], config_model["block_length"])    

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=config_train["batch_size"], 
    shuffle=False,
    pin_memory = True,
    num_workers = 0
)

def decode_sent(x, y):
    tokenizer_dir = Path.home() / "latin-literature-dataset-170M" / "tokenizer_model"
    last_tokens = 16
    in_decoded = [tokenizer_spm.decode_string(row[-last_tokens:].tolist(), tokenizer_dir) for row in x]

    ans_encoded = y
    ans_decoded = [tokenizer_spm.decode_string(row.tolist()[-1], tokenizer_dir) for row in ans_encoded]

    print(f"Input sentence was {in_decoded}")
    print(f"Last token then was {ans_decoded}")

for batch, (x, y) in enumerate(dataloader):
    decode_sent(x, y)
    input()