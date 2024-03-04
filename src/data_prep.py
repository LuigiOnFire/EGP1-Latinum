import json
from cltk.tokenizers import LatinTokenizationProcess
from cltk.languages.example_texts import get_example_text
from cltk.core.data_types import Doc

# check if the file is there
# later we can download it if it's not there
# for now just grab De Bello Gallico, later seek through each folder and grab every .xml.json
with open('../data/caes.bg_lat.xml.json', 'r') as file:
    data = json.load(file)
# # seperate/delimit chapters somehow?
text = ""
for book in data['TEI.2']['text']['body']['div1']:
    for chapter in book['p']:
            text += chapter['#text']
            text += '\n'

print(text)
# tokenize
tokenizer_process = LatinTokenizationProcess()
# print(Doc(raw=get_example_text("lat")[:23]))
tokenized_doc = tokenizer_process.run(input_doc=Doc(raw=text[0:50]))
print(tokenized_doc.tokens)
