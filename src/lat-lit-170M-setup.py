import csv
import numpy as np
from pathlib import Path

import data_prep
import tokenizer_spm

def unpack_csv_to_txts(txts_dir):
    # open the csv
    csv.field_size_limit(100000000) 
    repo_dir = Path.home() / "latin-literature-dataset-170M" 
    Path.mkdir(txts_dir, exist_ok=True)
    # have some expression ready to filter out test docs
    test_seneca = ""

    with open(repo_dir / "latin_raw.csv", newline='') as rawfile:

        # lemmas, word_count, lemma_count
        lemmas_reader = csv.reader(lemmasfile) 

        # title, text, author, year
        raw_reader = csv.reader(rawfile)

        lemmas_row = next(lemmas_reader)
        raw_row = next(raw_reader)

        # write each text in raw texts directory
        for row in raw_reader:
            title = (row[3] + "_" + row[1])
            if row[3] == "icero":
                title = "C" + title
            title = ''.join(title.split())
            title = title.replace("/", "_")
            title = title.replace(",", "_")
            filename = title + ".txt"

            with open(txts_dir / filename, "w") as titlesfile:
                lines = data_prep.sanitize_txt(row[2])
                titlesfile.write(lines)
                print(filename)

def train_tokenizer(txts_dir, tokenizer_model_dir):
    Path.mkdir(tokenizer_model_dir, exist_ok=True)

    txt_list = []
    for file in txts_dir.iterdir():
        txt_list.append(file)

    tokenizer_spm.train_tokenizer_bpe(txt_list, tokenizer_model_dir)

def make_training_bins(txts_dir, tokenizer_model_dir, train_bin_dir, test_bin_dir):
    Path.mkdir(train_bin_dir, exist_ok=True)
    Path.mkdir(test_bin_dir, exist_ok=True)

    train_bin = train_bin_dir / "train_encoded.bin"
    test_bin = test_bin_dir / "test_encoded.bin"

    # should delete thes if they already exist
    Path.unlink(train_bin, missing_ok=True)
    Path.unlink(test_bin, missing_ok=True)

    # initialize the memmaps
    np.memmap(train_bin, dtype=np.uint16, mode='w+', shape=(1,))
    np.memmap(test_bin, dtype=np.uint16, mode='w+', shape=(1,))

    def append_to_bin(encoded_str, bin, ln):
        str_len = len(encoded_str)
        ln += str_len
        memmap = np.memmap(bin, dtype=np.uint16, mode='r+', shape=(ln,))
        memmap[-str_len:] = encoded_str

        return ln
        
    train_ln = 0
    test_ln = 0
    for src_txt in txts_dir.iterdir():        
        name = src_txt.stem
        print(f"Tokenizing: {name}")
        encoded_str = tokenizer_spm.encode_doc(src_txt, tokenizer_model_dir)
        if name == "Seneca_AdLuciliumepistulaemorales":
            test_ln = append_to_bin(encoded_str, test_bin, test_ln)
            print(f"Saved {name} as test instead of training data")

        else:
            train_ln = append_to_bin(encoded_str, train_bin, train_ln)

if __name__ == "__main__":
    txts_dir = Path.home() / "latin-literature-dataset-170M" / "raw_txts"
    tokenizer_model_dir = Path.home() / "latin-literature-dataset-170M" / "tokenizer_model"
    train_bin_dir = Path.home() / "latin-literature-dataset-170M" / "bin_train"
    test_bin_dir = Path.home() / "latin-literature-dataset-170M" / "bin_test"

    # unpack_csv_to_txts(txts_dir)

    # tokenize with all the txt, we've set sampling to 1e6
    # save the tokenizer in its own directory
    # train_tokenizer(txts_dir, tokenizer_model_dir)

    # write the training bin files to a train folder and the testing bins to a test folder
    make_training_bins(txts_dir, tokenizer_model_dir, train_bin_dir, test_bin_dir)

    # hopefully we can use our old dataloader? if not make a new one
