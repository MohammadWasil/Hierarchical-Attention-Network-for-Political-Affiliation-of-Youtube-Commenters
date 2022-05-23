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
    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/3. scrap_youtube_twitter_handle.csv"))
    channel_yt_twitter.drop("index",axis=1,inplace=True)

    # duplicate one column
    channel_yt_twitter['Youtube User'] = channel_yt_twitter['Youtube Channel'] 

    #To find the id
    part = "id"

    # iterate over the dataframe
    for index, row in channel_yt_twitter.iterrows():
        
        channel_id_user = channel_yt_twitter.iloc[index, 4]
        
        if not pd.isna(channel_id_user):
            
            # first, we will call get_youtube_channel_id().
            channel_id_response = get_youtube_channel_id(youtube_service, part, channel_id_user)
            print(channel_id_response)

            if channel_id_response == "":
                print("Cannot find from the first function, trying again from the second function!!!")
                channel_id_response = check_youtube_channel_id(youtube_service, part, channel_id_user)
                print(channel_id_response)
                if channel_id_response == "":
                    print("Cannot find from the second function!!!")
                else:
                    channel_yt_twitter.iloc[index, 4] = channel_id_response
            else:
                channel_yt_twitter.iloc[index, 4] = channel_id_response



    save_csv_file = "data/5. scrap_youtube_ids.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")


if __name__ == '__main__':
    main()