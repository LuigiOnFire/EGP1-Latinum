from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc
import json
import numpy as np
from pathlib import Path


def download_perseus_data():
    # This is an important functionality but we'll pass on it for now
    pass

def perseus_json_to_string(filepath):
    # check if the file is there
    # later we can download it if it's not there
    # for now just grab De Bello Gallico, later seek through each folder and grab every .xml.json
    with open(filepath, 'r') as file:
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

def tokenize_data(docstring, data_dir):
    tokens = docstring_to_tokens(docstring)

def make_encoder(tokenized_docs, data_dir):
    all_tokens = np.array(set([token for doc in all_tokens for token in doc]))
    decoder = { index:token for index,token in enumerate(all_tokens) } 
    encoder = { index:token for index,token in enumerate(all_tokens) } 

    # let's use jsons to start for readability/validation but use pickle later for efficiency
    decoder_file = data_dir / "decoder.json"
    encoder_file = data_dir / "encoder.json"

    with open(decoder_file, 'w') as handle:
        json.dump(decoder, handle)

    with open(encoder_file, 'w') as handle:
        json.dump(encoder, handle)
    
def save_tokenized_doc()
    token_ids = np.array(tokens, dtype=np.uint16)
    token_ids_file = data_dir / "token_ids.bin"
    token_ids.tofile(os.path.join(token_ids_file), "../data/tokens.bin")


def prep_data():
    # if there is no perseus folder, or it's empty
    # download_perseus_data()
    batch_size = 0
    data_dir = Path(__file__).parent.parent / "data"

    # filepaths = ['../data/caes.bg_lat.xml.json']
    filepaths = get_json_docs(data_dir)
    tokenized_docs = []

    print("Entering loop")
    for filepath in filepaths:
        docstring = perseus_json_to_string(filepath)

        print(f"Tokenizing: {docstring}")
        tokenized_docs.append(tokenize_data(docstring, data_dir))

    make_encoder(tokenized_docs, data_dir)
    for doc in tokenized_docs:
        save_tokenized_doc(doc)

