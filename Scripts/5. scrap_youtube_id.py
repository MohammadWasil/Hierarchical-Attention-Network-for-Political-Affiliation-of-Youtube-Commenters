"""
Time = < 5 mins.
To validate the given youtube channel Id. If youtube channel username is available to us, we will get their youtube channel ID.

Input : 4. scrap_youtube_channel.csv
Output : 5. scrap_youtube_ids.csv
         5. list_indices_updated_yt_channel.txt (It contains indices which were updated in the csv file)
"""

import os
import pandas as pd
import yaml

from utils import Authenticate_youtube, get_youtube_channel_id, check_youtube_channel_id

def main():
    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

    # load config_KEYS.yaml file
    with open(os.path.join(DIRECTORY_PATH, "utility/config_KEYS.yml"), "r") as ymlfile:
        cfg =  yaml.safe_load(ymlfile)

    youtube_service = Authenticate_youtube(cfg)

    ################ CHANGE THIS #######################s
    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/4. scrap_youtube_channel.csv"))

    # duplicate one column
    channel_yt_twitter['Youtube User'] = channel_yt_twitter['Youtube Channel'] 

    #To find the id
    part = "id"

    # list of indices changed
    list_indices_updated = []

    # iterate over the dataframe
    for index, _ in channel_yt_twitter.iterrows():
        
        channel_id_user = channel_yt_twitter.iloc[index, 4]
        
        if not pd.isna(channel_id_user):
            print("-------------------- Youtube Channel Found FROM CSV -------------------- ")    
            # first, we will call get_youtube_channel_id().
            channel_id_response = get_youtube_channel_id(youtube_service, part, channel_id_user)
            
            if channel_id_response == "":
                print("Cannot find from youtube channel USERNAME, trying again from youtube channel ID!!!")
                channel_id_response = check_youtube_channel_id(youtube_service, part, channel_id_user)
                
                if channel_id_response == "":
                    print("Cannot find from youtube channel ID!!!")
                else:
                    print("ID: ", channel_id_user, " | ID Received: ", channel_id_response)
                    print("Matched using youtube channel ID")
                    channel_yt_twitter.iloc[index, 4] = channel_id_response

                    list_indices_updated.append(index)
            else:
                print("USER: ", channel_id_user, " | ID Received: ", channel_id_response)
                print("Matched using youtube channel USERNAME")
                channel_yt_twitter.iloc[index, 4] = channel_id_response

                list_indices_updated.append(index)
            print("\n")
        else:
            print("-------------------- No Youtube Channel Found From CSV -------------------- ")
            print("\n")

    print("Number of updated indices - {}".format(len(list_indices_updated)))
    
    print("Saving list ...")
    save_list = "data/5. list_indices_updated_yt_channel.txt"
    with open(os.path.join(DIRECTORY_PATH, save_list), "w") as output:
        output.write(str(list_indices_updated))
    print("List saved!")

    save_csv_file = "data/5. scrap_youtube_ids.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")


if __name__ == '__main__':
    main()