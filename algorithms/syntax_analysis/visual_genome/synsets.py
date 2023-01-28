# an Image driven tutorial is available here : http://visualgenome.org/api/v0/api_beginners_tutorial.html
# This file is a data exploration
import json

with open('../../../data/visual_genome/synsets.json') as json_file:
    data = json.load(json_file)

    for i, o in enumerate(data):
        print(f" name {o['synset_name']} definition {o['synset_definition']}")

        if i == 1000:
            break
