import json
import re

from cltk.alphabet import lat
from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc
from cltk.data.fetch import FetchCorpus
from cltk.sentence.lat import LatinPunktSentenceTokenizer

from datasets import load_dataset
import numpy as np
import os # TODO: Can we make everything use pathlib instead?
from pathlib import Path

import data_filenames
import tokenizer_spm
import utils
from utils import PAD_TOKEN, PAD_INDEX

def download_perseus_data():
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.list_corpora
    corpora = ["lat_text_perseus"]
    for corpus in corpora:
        corpus_downloader.import_corpus(corpus)

def download_lat_tesserae_data():
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.list_corpora
    corpora = ["lat_text_tesserae"]
    for corpus in corpora:
        corpus_downloader.import_corpus(corpus)

def download_lat_library_data():
    corpus_downloader = FetchCorpus(language="lat")
    corpus_downloader.list_corpora
    corpora = ["lat_text_latin_library"]
    for corpus in corpora:
        corpus_downloader.import_corpus(corpus)
    
"""
These next few functoins are for parsing the .xaml.jsons
We're not currently using them, but they'll be useful later on if we want more data.
"""
def get_p_text(data, display=False):
    def process_text(ele):
        newline = ""
        if '#text' in ele.keys():
            newline = ele['#text']

        elif 'quote' in ele.keys():
            newline = ele['quote']['#text']

        newline = re.sub(r"[\n]+", " ", newline)

        # make sure the new texts aren't running into each other
        # I find seperating with a new line more readable
        newline += "\n" 
        if display:
            print(newline)
            input()

        return newline

    newline = ""
    # sometimes p is in array of dicts
    # sometimes p directly contains a single dict
    # we accomodate for both
    if isinstance(data['p'], dict):
        newline = process_text(data['p'])

    elif isinstance(data['p'], list):
        for p in data['p']:
            newline += process_text(p)

    return newline

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
            newline = get_cltk_json_node_text(data[key], display)
            
            text += newline
    
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
        data = json.load(file)
    # seperate/delimit chapters somehow?
    text =  parse_cltk_json(data, display=False)

    return text

def tess_to_lines(filepath):
    # check if the file is there
    # later we can download it if it's not there 
    with open(filepath, 'r', encoding="utf8") as file:
        lines = file.readlines()
            
    for i, line in enumerate(lines):
        bracket_pattern = "^<.+?>"
        lines[i] = re.sub(bracket_pattern, "", line)

    return lines

def sanitize_txt(text):
    def sanitize_line(line):
        linenum_pattern = r"\s{3}\d+"

        bracket_num_pattern = r"\[[\d\-]+\]"

        bracket_pattern = r"[\[\]]"

        newline = line
        newline = re.sub(linenum_pattern, "", newline)

        # first get rid of the brackets containing numbers (line/verse labels etc)
        newline = re.sub(bracket_num_pattern, "", newline)

        # then just get rid of all the remaining brackets
        newline = re.sub(bracket_pattern, "", newline)

        newline = newline.lower()

        # doing this makes me sad but I think it's for the better
        # hopefully in a future version of EGP it won't be necessary!
        newline = newline.replace('v', 'u')
        newline = newline.replace('j', 'i')

        return newline

    if isinstance(text, str):
        text = sanitize_line(text)

        return text

    # TODO: fix this silly silly dry violation
    elif isinstance(text, list):
        newlines = text
        for i, line in enumerate(text):    
            newlines[i] = sanitize_line(line)            

        return newlines

def lib_to_lines(filepath):
    # check if the file is there
    # later we can download it if it's not there 
    # apparently lib files are not utf-8 encoded and need to be for sentence piece
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # take off the title since it's in english sometimes
    # or I'd like too but some files are one line
    if len(lines) > 1:
        lines = lines[1:]
    
    latlib_pattern = r"The Latin Library"

    if len(lines) > 1 and re.search(latlib_pattern, lines[-1]):
        lines = lines[:-1]

    elif len(lines) > 3 and re.search(latlib_pattern, lines[-2]):
        lines = lines[:-3]

    newlines = lines

    for i, line in enumerate(lines):
        newlines[i] = line.replace('\0', '')
    
    lines = newlines

    lines = sanitize_txt(lines)

    return lines


def save_docstring(docstring, filename, token_data_dir):        
    extension = ".txt"
    filename = filename.replace("/", "-")
    output_path = token_data_dir / (filename + extension)
    
    # need this for 
    output_path.parent.mkdir(parents=True, exist_ok=True)    

    # make the document all lower for easier tokenization
    with open(output_path, "w", encoding="utf-8") as f:
        if isinstance(docstring, str):
            docstring = docstring.lower()
            f.write(docstring)
        elif isinstance(docstring, list):
            for string in docstring:
                string = string.lower()

            f.writelines(docstring)
                
    return output_path

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

# TODO: can delete this later once get_json_docs is implemented
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

def get_json_doc(src_filename, source_data_dir):
    extension = ".json"
    filepath = source_data_dir / (src_filename + extension)

    return filepath

def get_tess_file(filename, source_data_dir):
    for item in source_data_dir.iterdir():
        if item.stem.startswith(filename):          
            return item

def get_lib_file(filename, source_data_dir):
    split_name = filename.split("/")
    if len(split_name) == 1:
        for item in source_data_dir.iterdir():
            if item.stem.startswith(filename):          
                return item

    elif len(split_name) == 2:
        for item in (source_data_dir / split_name[0]).iterdir():
            if item.stem.startswith(split_name[1]):          
                return item


def encode_doc(tokenized_doc, encoder):
    encoded_doc = []
    for token in tokenized_doc:
        encoded_doc.append(encoder[token])

    return encoded_doc

def save_encoded_doc(encoded_doc, name, token_data_dir):
    # let's keep them in the same directory    
    name = name.replace("/", "-")
    token_ids = np.array(encoded_doc, dtype=np.int64)
    name = name + "_encoded.bin"
    token_ids_file = token_data_dir / name    

    token_ids.tofile(token_ids_file)


def json_to_txt(filenames, src_dir, token_data_dir):
    # sorting for prose
    txt_filepaths = []

    for filename in filenames:
        src_filepath = get_json_doc(filename, src_dir)
        docstring = perseus_json_to_string(src_filepath)
        txt_filepath = save_docstring(docstring, filename, token_data_dir)
        txt_filepaths.append(txt_filepath)
    
    return txt_filepaths

def tess_to_txt(filenames, src_dir, token_data_dir):
    txt_filepaths = []

    for filename in filenames:
        src_filepath = get_tess_file(filename, src_dir)        
        doclines = tess_to_lines(src_filepath)
        txt_filepath = save_docstring(doclines, filename, token_data_dir)
        txt_filepaths.append(txt_filepath)
    
    return txt_filepaths

def lib_to_txt(filenames, src_dir, token_data_dir):
    txt_filepaths = []

    for filename in filenames:
        src_filepath = get_lib_file(filename, src_dir)        
        doclines = lib_to_lines(src_filepath)
        txt_filepath = save_docstring(doclines, filename, token_data_dir)
        txt_filepaths.append(txt_filepath)
    
    return txt_filepaths

# this is kinda janky, if we want to get perseus working at the same time 
# we need to do some legwork
def current_works():
    filenames = []
    perseus_filenames = []
    # perseus_filenames = data_filenames.get_perseus_filenames()

    # tessera_filenames = data_filenames.get_tessera_filenames()
    lat_lib_train_filenames = data_filenames.get_lat_lib_train_filenames()
    filenames += lat_lib_train_filenames

    return lat_lib_train_filenames, filenames

def prep_data(model_dir):
    """
    This is the top level function of this file
    """

    perseus_cltk_json_data_dir = utils.perseus_cltk_json_data_dir
    lat_tesserae_data_dir = utils.lat_library_data_dir
    lat_library_data_dir = utils.lat_library_data_dir

    token_data_dir = utils.token_data_dir

    test_data_dir = utils.test_data_dir    

    # if there is no perseus folder, or it's empty
    if not perseus_cltk_json_data_dir.exists():
        download_perseus_data(perseus_cltk_json_data_dir)

    if not lat_tesserae_data_dir.exists():
        download_lat_tesserae_data()

    if not lat_library_data_dir.exists():
        download_lat_library_data()

    # Clear the directory of its contents if it exists
    if token_data_dir.exists():
        utils.empty_dir(token_data_dir)    
   
    # Make the directory if it doesn't exist yet
    Path.mkdir(token_data_dir, exist_ok=True)   
    Path.mkdir(test_data_dir, exist_ok=True)

    lat_lib_train_filenames, filenames = current_works()

    lat_lib_test_filenames = data_filenames.get_lat_lib_test_filenames()

    # not sure I need these anymore
    # txt_filepaths = json_to_txt(perseus_filenames, perseus_cltk_json_data_dir, token_data_dir)

    # get file paths from all tessera files
    # txt_filepaths_train = tess_to_txt(tessera_filenames, lat_tesserae_data_dir, token_data_dir)
    # txt_filepaths_test = tess_to_txt(test_filenames, lat_tesserae_data_dir, token_data_dir)

    txt_filepaths_train = lib_to_txt(lat_lib_train_filenames, lat_library_data_dir, token_data_dir)
    txt_filepaths_test = lib_to_txt(lat_lib_test_filenames, lat_library_data_dir, token_data_dir)

    txt_filepaths_to_tokenize = txt_filepaths_train + txt_filepaths_test
    tokenizer_spm.train_tokenizer_bpe(txt_filepaths_to_tokenize, model_dir)

    for ind, txt_filepath in enumerate(txt_filepaths_train):
        encoded_str = tokenizer_spm.encode_doc(txt_filepath, model_dir)
        save_encoded_doc(encoded_str, lat_lib_train_filenames[ind], token_data_dir)

    for ind, txt_filepath in enumerate(txt_filepaths_test):
        encoded_str = tokenizer_spm.encode_doc(txt_filepath, model_dir)
        save_encoded_doc(encoded_str, lat_lib_test_filenames[ind], test_data_dir)

    # this is passsed as a (crude) measure of a model's complexity
    # to use in the serial number
    return filenames

# TODO: this is a super violation of DRY, refactor
# using the above later, but I want to start getting test loss
# samples ASAP
def prep_test_data(model_dir): # need model dir just for tokenizer
    """
    Similar to above but for test data alone
    """
    # TODO: move these to utils
    lat_tesserae_data_dir = Path.home() / "cltk_data" / \
        "lat" / "text" / "lat_text_tesserae" / "texts"
    
    lat_library_data_dir = Path.home() / "cltk_data" / \
        "lat" / "text" / "lat_text_latin_library"
    
    test_data_dir = utils.test_data_dir

    if not lat_tesserae_data_dir.exists():
        download_lat_tesserae_data()

    if not lat_library_data_dir.exists():
        download_lat_library_data()

    filenames = data_filenames.get_test_filenames()

    # otherwise we might as well do all this over again
    # Clear the directory of its contents if it exists
    if test_data_dir.exists():
        utils.empty_dir(test_data_dir)

    all_exist = True
    for filename in filenames:
        bin_path = test_data_dir / (filename + ".bin")
        if not bin_path.exists():
            all_exist = False
            break
    
    # if all the bins already exist, we don't need to go through all this trouble
    if all_exist:
        return                    
   
    # Make the directory if it doesn't exist yet
    Path.mkdir(test_data_dir, exist_ok=True)   

    # get file paths from all test files
    txt_filepaths = tess_to_txt(filenames, lat_tesserae_data_dir, test_data_dir)

    for ind, txt_filepath in enumerate(txt_filepaths):
        encoded_str = tokenizer_spm.encode_doc(txt_filepath, model_dir)
        save_encoded_doc(encoded_str, filenames[ind], test_data_dir)

    # later let's make sure we do this in data_prep and return the file names
    # to save in the model_dir