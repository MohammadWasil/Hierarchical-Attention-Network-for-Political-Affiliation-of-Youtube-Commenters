"""
Another way to get the channel id by using youtube search method

Input: data/5.1 get_yt_ids.csv
Output: data/5.1 yt_ids_update_idx_list.txt
        data/5.1 get_yt_ids.csv
"""

import os
import pandas as pd
import yaml
from collections import defaultdict
import requests
from difflib import SequenceMatcher
from utils import Authenticate_youtube

def yt_similarity(id, channel_yt_twitter, youtube_service):
    response = youtube_service.channels().list(part = "brandingSettings, contentDetails, contentOwnerDetails, id ,localizations ,snippet, statistics, status, topicDetails", id=id)
    response = response.execute()

    selected_youtube_channel = ""
    if "items" in response:
        for item in response["items"]:
            if "snippet" in item:
                if "title" in item["snippet"]:
                    selected_youtube_channel = item["snippet"]["title"]
                elif "localized" in item["snippet"]:
                    if "title" in item["snippet"]["localized"]:
                        selected_youtube_channel = item["snippet"]["localized"]["title"]
            elif "brandingSettings" in item:
                if "channel" in item["brandingSettings"]:
                    if "title" in item["brandingSettings"]["channel"]:
                        selected_youtube_channel = item["brandingSettings"]["channel"]["title"]

    if selected_youtube_channel != "":
        yt_similarity = SequenceMatcher(None, channel_yt_twitter.loc[0, "Name"], selected_youtube_channel).ratio()
        return yt_similarity
    return 0


def main():
    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

    # load config_KEYS.yaml file
    with open(os.path.join(DIRECTORY_PATH, "utility/config_KEYS.yml"), "r") as ymlfile:
        cfg =  yaml.safe_load(ymlfile)

    youtube_service = Authenticate_youtube(cfg)

    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/5.1 get_yt_ids.csv"))
    # list of indices changed
    list_indices_updated = []

    # 222:
    for index, row in channel_yt_twitter.iterrows():

        channel_username = channel_yt_twitter.loc[index, "Youtube Channel"]
        if not pd.isna(channel_username):
            #channel_username = "National Youth Rights Association"
            channel_id = requests.get('https://www.googleapis.com/youtube/v3/search?part=id&q={}&type=channel&key={}'.format(channel_username, cfg["YOUTUBE_KEY"])).json()

            yt_channel_id = []

            if "items" in channel_id:
                print(len(channel_id["items"]))

                for item in channel_id["items"]:
                    if "id" in item:
                        if "channelId" in item["id"]:
                            id_ = item["id"]["channelId"]
                            yt_channel_id.append(id_)
                if len(yt_channel_id) > 0:
                    print("Index: ", index)
                    print(channel_username, " - ", yt_channel_id[0])    
                    # update the csv file
                    channel_yt_twitter.loc[index, "Youtube Channel"] = yt_channel_id[0]
                    list_indices_updated.append(index)

    print("Saving list ...")
    save_list = "data/5.1 yt_ids_update_idx_list.txt"
    with open(os.path.join(DIRECTORY_PATH, save_list), "w") as output:
        output.write(str(list_indices_updated))
    print("List saved!")


    save_csv_file = "data/5.1 get_yt_ids.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")

if __name__ == '__main__':
    main()