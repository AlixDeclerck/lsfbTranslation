# an Image driven tutorial is available here : http://visualgenome.org/api/v0/api_beginners_tutorial.html
# This file is a data exploration
import json

with open('../../../data/visual_genome/relationships.json') as json_file:
    data = json.load(json_file)

    for i, d in enumerate(data):
        for j, r in enumerate(d["relationships"]):
            print(f" name {r['object']['name']} synsets {r['object']['synsets']}")

        if i == 1000:
            break
