from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc
import json
import numpy as np
from pathlib import Path

def download_perseus_data():
    # This is an importnat functionality but we'll pass on it for now
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
    tokenized_doc = tokenizer_process.run(input_doc=Doc(raw=text[0:50]))
    return tokenized_doc.tokens

def tokenize_data(docstring):
    tokens = docstring_to_tokens(docstring)
    # we also need to encode the tokens but we'll do that later
    token_ids = tokens
    token_ids = np.array(tokens, dtype=np.uint16)
    data_dir = pathlib.Path(__file__).resolve.parent.parent / "data"
    token_ids_file = data_dir /"token_ids.bin"
    token_ids.tofile(os.path.join(token_ids_file), "../data/tokens.bin"))


def prep_data():
    # if there is no perseus folder, or it's empty
    # download_perseus_data()
    filepath = '../data/caes.bg_lat.xml.json'
    docstring = perseus_json_to_string(filepath)

    tokenize_data(docstring)

