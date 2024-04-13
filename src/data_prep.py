import json
import re

from cltk.alphabet import lat
from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc
import numpy as np
import os # TODO: Can we make everything use pathlib instead?
from pathlib import Path
import utils
from utils import PAD_TOKEN, PAD_INDEX


def download_perseus_data():
    # This is an important functionality if we want quick setup on other machines
    # To implement later on
    pass

"""
These next few functoins are for parsing the .xaml.jsons
We're not currently using them, but they'll be useful later on if we want more data.
"""
def get_p_text(data, display=False):
    def process_text(ele):
        newtext = ""
        if '#text' in ele.keys():
            newtext = ele['#text']

        elif 'quote' in ele.keys():
            newtext = ele['quote']['#text']

        newtext = re.sub(r"[\n]+", " ", newtext)


        # make sure the new texts aren't running into each other
        # I find seperating with a new line more readable
        newtext += "\n" 
        if display:
            print(newtext)
            input()

        return newtext

    newtext = ""
    # sometimes p is in array of dicts
    # sometimes p directly contains a single dict
    # we accomodate for both
    if isinstance(data['p'], dict):
        newtext = process_text(data['p'])

    elif isinstance(data['p'], list):
        for p in data['p']:
            newtext += process_text(p)

    return newtext

def get_div_text(data, n, display=False):
    text = ""
    if f"div{n}" in data.keys():
        for i, section in enumerate(data[f"div{n}"]):
            if display: 
                print(f"div{n}: {i}") 
            text += get_div_text(section, n + 1, display=display)
    
    elif "p" in data.keys():
        text += get_p_text(data, display=display)
    
    return text

# The display flag is for debugging
def parse_te2_json(data, display=False):
    text = ""
    text += get_div_text(data['TEI.2']['text']['body'], 1, display=display)
    return text

def get_cltk_json_node_text(data, display=False):
    text = ""
    if isinstance(data, dict):
        for key in data:
            newtext = get_cltk_json_node_text(data[key], display)
            
            text += newtext
    
    elif isinstance(data, str):
        data = re.sub(r"[\t\n]+", " ", data)
        text += data
        if display:
            print(data)
            input()
    
    return text

def parse_cltk_json(data, display=False):
    return get_cltk_json_node_text(data["text"], display=display)

def perseus_json_to_string(filepath):
    # check if the file is there
    # later we can download it if it's not there 
    with open(filepath, 'r', encoding="utf8") as file:
        print(filepath)
        data = json.load(file)
    # seperate/delimit chapters somehow?
    text =  parse_cltk_json(data, display=False)

    return text

def docstring_to_tokens(docstring):
    # tokenize
    docstring = docstring.lower()
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
    extension = ".json"
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
    name = name + "_encoded.bin"   

    token_ids_file = token_data_dir / name    

    token_ids.tofile(token_ids_file)

def prep_data():
    """
    This is the top level function of this file
    """
    # if there is no perseus folder, or it's empty
    # download_perseus_data()
    # TODO: move these to utils
    source_data_dir = Path(__file__).parent.parent / "data" / "cltk_json"
    token_data_dir = Path(__file__).parent.parent / "token_data"

    # Clear the directory of its contents
    utils.empty_dir(token_data_dir)

    # Make the directory if it doesn't exist yet
    os.makedirs(token_data_dir, exist_ok=True)

    
    filenames = [
        "caesar-julius__gallic-war__latin",
        "apuleius__metamorphoses__latin",
        # "celsus-aulus-cornelius__de-medicina__latin",
        # "cicero__academica__latin",
        # "cicero__against-publius-servilius-rullus__latin",
        # "cicero__brutus__latin",
        # "cicero__de-amicitia__latin",
        # "cicero__de-divinatione__latin",
        # "cicero__de-fato__latin",
        "cicero__de-finibus-bonorum-et-malorum__latin",
        # "cicero__de-inventione__latin",
        # "cicero__de-natura-deorum__latin",
        "cicero__de-officiis__latin"
        # "cicero__de-optimo-genere-oratorum__latin",
        # "cicero__de-republica__latin"
        ]
    filepaths = get_json_docs(filenames, source_data_dir)
    tokenized_docs = []

    for filepath in filepaths:
        docstring = perseus_json_to_string(filepath)

        tokenized_docs.append(docstring_to_tokens(docstring))

    _, encoder = make_encoder(tokenized_docs, token_data_dir)
    for ind, name in enumerate(filenames):
        encoded_doc = encode_doc(tokenized_docs[ind], encoder)
        save_encoded_doc(encoded_doc, name, token_data_dir)

