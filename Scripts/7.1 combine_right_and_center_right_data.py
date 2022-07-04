import json

DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

# combine comments from RIGHT and CENTER RIGHT
input_file = "data/7. comments CENTRE RIGHT.json" # take data from here
output_file = "data/7. comments RIGHT.json" # and save it here

with open(input_file, 'r') as fin:
    with open(output_file, 'a') as fout:
        for line in fin:
            line_in = json.loads(line.rstrip())
            fout.write('{0}\n'.format(json.dumps(line_in)))