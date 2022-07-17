"""
First step of annotation is taking place here.
Annotation is being done using subscription list of the users/authors, and homogeneity score.
"""

import pandas as pd
import json, os
from utils import homogeneity_score, biasness_classification

DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

def main():
    print("Reading video data ...")
    ## common  for LEFT and RIGHT
    input_file = "data/6. video_ids.json"
    LEFT_YT_channels = []
    with open(os.path.join(DIRECTORY_PATH, input_file), 'r') as fin:
        for line in fin:
            l = json.loads(line.rstrip())
            LEFT_YT_channels.append(l["channel id"])
            
    input_file = "data/6. video_ids_ RIGHT.json"
    RIGHT_YT_channels = []
    with open(os.path.join(DIRECTORY_PATH, input_file), 'r') as fin:
        for line in fin:
            l = json.loads(line.rstrip())
            RIGHT_YT_channels.append(l["channel id"])

    # change this for "LEFT" or "RIGHT" leaning channels data
    LEANING = "LEFT"

    print("Creating users leaning ...")
    subscription_list = "data/9. authors_subscription {}.json".format(LEANING)
    users_leaning = {}
    with open(os.path.join(DIRECTORY_PATH, subscription_list), 'r') as fin:
        for line in fin:
            l = json.loads(line.rstrip())
            user = list(l.keys())[0]
            subscribed_channels = list(list(l.values())[0].keys())
            left = 0
            right = 0
            for sc in subscribed_channels:
                if sc in LEFT_YT_channels:
                    left += 1
                elif sc in RIGHT_YT_channels:
                    right += 1
            if right != 0 or left != 0:
                homo_score = homogeneity_score(right, left)
                users_leaning[user] = homo_score

    print("Total number of users leaning found: {}".format(len(users_leaning)))

    # update the homogeneity score in the csv file
    channel_csv = "data/8. comments {}.csv".format(LEANING)
    channels = pd.read_csv(os.path.join(DIRECTORY_PATH, channel_csv))

    print("Updating DataFrame ...")
    # create a new column in the dataframe
    channels["Homogeneity Score"] = -1000
    channels["Authors Biasness"] = ""

    for key, value in users_leaning.items():
        indx = channels[channels["Author Id"] == key].index

        channels.loc[indx, "Homogeneity Score"] = value
        channels.loc[indx, "Authors Biasness"] = biasness_classification(value)

    save_csv_file = "data/10. comments {}.csv".format(LEANING)
    channels.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")

if __name__ == '__main__':
    main()