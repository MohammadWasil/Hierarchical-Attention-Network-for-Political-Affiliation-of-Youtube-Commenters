"""
From the youtube channel's playlist, get all those videos which was published from 2021-01-01 to 2021-08-31. 

Input: 5.1 get_yt_ids.csv
Output: 6. video_ids.json
"""

import pandas as pd
import os, json
import pandas as pd
import time
from utils import get_video_ids_playlist

def main():
    start = time.time()
    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"
    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/5.1 get_yt_ids.csv"))
    
    output_file = "data/6. video_ids.json"

    visited_channel_set = set()
    
    if os.path.exists(os.path.join(DIRECTORY_PATH, output_file)):
        with open(os.path.join(DIRECTORY_PATH, output_file), 'r') as fin:
            for line in fin:
                visited_channel_set.add(json.loads(line.rstrip())['channel id'])
    num_of_channels_scraped = len(visited_channel_set)
    print('Already visited {0} channels before, let us continue ... '.format(num_of_channels_scraped))
    
    print(visited_channel_set)
    print("***************************************************************************************")

    with open(os.path.join(DIRECTORY_PATH, output_file), 'a') as fout:
        # iterate over the dataframe
        for index, _ in channel_yt_twitter[:956].iterrows():
            
            channel_id = channel_yt_twitter.loc[index, "Youtube Channel"]

            if not pd.isna(channel_id):
                if channel_id not in visited_channel_set: 
                    print("Processing index number: {}".format(index))
                    num_of_channels_scraped += 1

                    # to scrap all the videos information from a single channel,
                    video_ids = get_video_ids_playlist(channel_id)

                    fout.write('{}\n'.format(json.dumps({"channel id" : channel_id,
                                                        "video_ids" : video_ids})))
                    
                    visited_channel_set.add(channel_id)
                    print("***************************************************************************************")

    print("Number of channels scraped: ", num_of_channels_scraped)
    print("\n")
    print('It took {0:0.1f} seconds'.format(time.time() - start))

if __name__ == '__main__':
    main()