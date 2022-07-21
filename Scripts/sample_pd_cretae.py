import pandas as pd
import os
from utils import preprocess, dictionary, DIRECTORY_PATH
import json
data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/For inference sample.csv"))


left = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/13. comments RIGHT with Hashtag annotations.csv"))
left_unlabel = left[left['Authors Biasness'].isnull()]

ids = left_unlabel[0:9].groupby('Author Id')["Author Name"].value_counts().index[:]
save_ids = []

output_file = "data/Track the ids.txt"
if os.path.exists(os.path.join(DIRECTORY_PATH, output_file)):
        with open(os.path.join(DIRECTORY_PATH, output_file), 'r') as fin:
            for line in fin:
                save_ids.append(line[:-1])
            print(save_ids)
                #video_json = json.loads(fin.rstrip())
                
                #if 'channel id' in video_json:
                #    #for vid in video_json["video_ids"]:
                #    visited_channel.add(video_json['channel id'])
        #num_of_channels_scraped = len(visited_channel)

for i in ids:
    user_id = i[0]
    if user_id not in save_ids:
        right_comments = []
        right_num_of_comments = []
        
        #right_comments.append(R.loc[R['Author Id'] == user_id]["Authors Comment"].tolist())
        comments = left_unlabel.loc[left_unlabel['Author Id'] == user_id]["Authors Comment"]
        r_text = ""
        right_num_of_comment = 0
        for comment in comments:
            text = preprocess(comment, dictionary)
            r_text = r_text + text + " -|- "
            # count the number of comments in each documents.
            right_num_of_comment += 1
        
        right_comments.append(r_text)
        right_num_of_comments.append(right_num_of_comment)

        comments_left_channel = pd.DataFrame(
                {"comment" : right_comments,
                "Number of Comment" : right_num_of_comments
                })
        #comments_left_channel = [[right_comments, right_num_of_comments]] 
        #comments_left_channel = pd.DataFrame(comments_left_channel, columns = ['comment', 'Number of Comment']) 

        # Write the new data to the CSV file in append mode
        comments_left_channel.to_csv(os.path.join(DIRECTORY_PATH, 'data/For inference sample.csv'), mode='a', header=False, index=False)
        print('check test.csv')

        # save the id
        save_ids.append(i)
        with open(os.path.join(DIRECTORY_PATH, output_file), 'a') as fin:
            fin.write(f"{user_id}\n")