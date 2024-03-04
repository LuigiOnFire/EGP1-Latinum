import json

# Used to download the data, which we aleady have
# from cltk.data.fetch import FetchCorpus
# corpus_downloader = FetchCorpus(language='lat')
# print(corpus_downloader.list_corpora)
# corpus_downloader.import_corpus("lat_models_cltk")

with open('caes.bg_lat.xml.json', 'r') as file:
    data = json.load(file) 


for book in data['TEI.2']['text']['body']['div1']:
    for chapter in book['p']:
            print(chapter['#text'])
