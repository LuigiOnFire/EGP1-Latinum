import csv
from pathlib import Path

csv.field_size_limit(100000000) 
repo_dir = Path.home() / "latin-literature-dataset-170M" 
# output_dir = Path.home() / "latin-literature-dataset-170M" / "texts"
# Path.mkdir(output_dir)
with open(repo_dir / "latin_lemmas.csv") as csvfile:    
        
        reader = csv.reader(csvfile)    
        header = next(reader)

        print(header)

        for row in reader:
            title = (row[3] + "_" + row[1])
            if row[3] == "icero":
                title = "C" + title
            filename = title + ".txt"
            filename = ''.join(filename.split())

            print(filename)

            input()    
            
            # with open(output_dir / title, "w") as titlesfile:
            #     titlesfile.write(row[2])
            #     input()
