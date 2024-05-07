"""
Just a script used to get all of a certain author's works out of cltk_json.
It's very ad hoc right now, but I'm keeping it in case it's useful later.
It can always be modified / expanded later on.
"""

from pathlib import Path
import re

# datapath = Path.home() / "cltk_data" / \
#        "lat" / "text" / "lat_text_perseus" / "cltk_json"

datapath = Path.home() / "cltk_data" / \
       "lat" / "text" / "lat_text_latin_library"

def scrape_texts(dir):
    works_to_get = []
    author = "cicero"
    extension = ".txt"
    # lang = "latin"

    author = re.escape(author)
    extension = re.escape(extension)
    # lang = re.escape(lang)

    # auth_ext_pattern = fr"^{author}(.*){extension}"
    ext_pattern = fr"^(.*){extension}"
    # lang_pattern = fr"{lang}"

    for item in dir.iterdir():
        name = item.name
        if item.is_file() and \
            re.search(ext_pattern, name) is not None: # and \
            # re.search(lang_pattern, name) is not None:\
            works_to_get.append(item.stem)
        if item.is_dir():
            dir_contents = scrape_texts(item)
            for dir_file in dir_contents:
                works_to_get.append(item.stem + "/" + dir_file)

    return works_to_get

works_to_get = scrape_texts(datapath)
    
works_to_get.sort()

with open("latin_lib_files.txt", "w") as out:
    for work in works_to_get:
        out.write(f"\"{work}\",\n")
