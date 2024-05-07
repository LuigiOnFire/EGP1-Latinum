import numpy as np
import sentencepiece as spm

from utils import VOCAB_SIZE, PAD_INDEX

# class TokenizerLat_BPE_SPM:
#     def __init__(self):

    
def train_tokenizer_bpe(txt_filepaths, model_dir):
    # we gotta add pad tokens
    spm.SentencePieceTrainer.train(
        input = txt_filepaths, 
        model_prefix=model_dir / 'm', 
        vocab_size=VOCAB_SIZE, 
        max_sentence_length=500000, 
        pad_id=PAD_INDEX, 
        hard_vocab_limit=False,
        character_coverage=0.99975, # increased a bit since we're predicting unks, it's greek's fault
        model_type="unigram",
        input_sentence_size=1000000,
        shuffle_input_sentence=True
    )

def encode_string(string, model_dir):
    model_file = str(model_dir / 'm.model')
    sp = spm.SentencePieceProcessor(model_file=model_file)
    string = string.lower()

    encoded_str = sp.encode(string, out_type=int)

    return encoded_str


def encode_doc(txt_filepath, model_dir, add_bos=True, add_eos=True):
    with open(txt_filepath, "r") as f:
        txt_str = f.read()

    encoded_str = encode_string(txt_str, model_dir)

    return encoded_str

def decode_string(vec, model_dir):
    model_file = str(model_dir / 'm.model')
    sp = spm.SentencePieceProcessor(model_file)
    decoded_str = sp.decode(vec)

    return decoded_str