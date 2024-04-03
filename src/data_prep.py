import json
import os

from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc
import numpy as np
from pathlib import Path
from utils import PAD_TOKEN, PAD_INDEX



def download_perseus_data():
    # This is an important functionality but we'll pass on it for now
    pass

def perseus_json_to_string(filepath):
    # check if the file is there
    # later we can download it if it's not there
    # for now just grab De Bello Gallico, later seek through each folder and grab every .xml.json
    with open(filepath, 'r') as file:
        print(filepath)
        data = json.load(file)
    # seperate/delimit chapters somehow?
    text = ""
    for book in data['TEI.2']['text']['body']['div1']:
        for chapter in book['p']:
                text += chapter['#text']
                text += '\n'

    return text

def docstring_to_tokens(docstring):
    # tokenize
    tokenizer_process = LatinTokenizationProcess()
    tokenized_doc = tokenizer_process.run(input_doc=Doc(raw=docstring))
    return tokenized_doc.tokens

def make_encoder(tokenized_docs, token_data_dir):
    all_tokens = list(set([token for doc in tokenized_docs for token in doc]))
    all_tokens.insert(PAD_INDEX, PAD_TOKEN)
    decoder = { index:token for index,token in enumerate(all_tokens) } 
    encoder = { token:index for index,token in enumerate(all_tokens) } 

    # let's use jsons to start for readability/validation but use pickle later for efficiency
    decoder_file = token_data_dir / "decoder.json"
    encoder_file = token_data_dir / "encoder.json"

    with open(decoder_file, 'w') as handle:
        json.dump(decoder, handle)

    with open(encoder_file, 'w') as handle:
        json.dump(encoder, handle)

    return decoder, encoder
    
def get_json_docs(seek_files, source_data_dir):
    extension = ".xml.json"
    filepaths = []
    for root, _, found_files in os.walk(source_data_dir):
        for seek_file in seek_files: # these are the sought filenames
            for found_file in found_files: # there are the files on disc
                if seek_file + extension == found_file:
                    filepaths.append(os.path.join(root, found_file))
                    break
    
    return filepaths

def encode_doc(tokenized_doc, encoder):
    encoded_doc = []
    for token in tokenized_doc:
        encoded_doc.append(encoder[token])

    return encoded_doc

def save_encoded_doc(encoded_doc, name, token_data_dir):
    token_ids = np.array(encoded_doc, dtype=np.int64)
    name = name + "encoded.bin"
    token_ids_file = token_data_dir / name
    token_ids.tofile(token_ids_file)

def prep_data():
    """
    This is the top level function of this file
    """
    # if there is no perseus folder, or it's empty
    # download_perseus_data()
    batch_size = 0
    source_data_dir = Path(__file__).parent.parent / "data"
    token_data_dir = source_data_dir / "token_data"

    # Make the directory if it doesn't exist yet
    os.makedirs(token_data_dir, exist_ok=True)

    filenames = ["caes.bg_lat"]
    filepaths = get_json_docs(filenames, source_data_dir)
    print(f"Filpeaths is {filepaths}")
    tokenized_docs = []

    for filepath in filepaths:
        print(f"the file path is {filepath}")
        docstring = perseus_json_to_string(filepath)

        tokenized_docs.append(docstring_to_tokens(docstring))

    _, encoder = make_encoder(tokenized_docs, token_data_dir)
    for ind, name in enumerate(filenames):
        encoded_doc = encode_doc(tokenized_docs[ind], encoder)
        save_encoded_doc(encoded_doc, name, token_data_dir)

