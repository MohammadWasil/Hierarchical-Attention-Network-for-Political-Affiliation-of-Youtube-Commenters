"""
Maually review the twitter handle. If the the similarity score of twitter handle is less than 0.5, manaully review and change them accordingly.

Input: 2. scrap_youtube_twitter.csv
Output: 2_1. review_twitter_handle.csv
"""

import pandas as pd
import os

# get all the twitter handles with similarity less than 0.5
# review_twitter_handle = channel_yt_twitter[channel_yt_twitter["Twitter Similarity"] < 0.5]
# Done manually

def main():
    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

    file_path = "data/2. scrap_youtube_twitter.csv"
    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, file_path))

    # santasurfing - account suspended
    # 2885 ReedMCooper - accoutn suspended
    channel_yt_twitter.loc[23, "Twitter Handle"] = ""
    channel_yt_twitter.loc[29, "Twitter Handle"] = ""
    channel_yt_twitter.loc[79, "Twitter Handle"] = "drudgeretort"
    channel_yt_twitter.loc[93, "Twitter Handle"] = "FifthEstateMag"
    channel_yt_twitter.loc[112, "Twitter Handle"] = "GQMagazine"
    channel_yt_twitter.loc[120, "Twitter Handle"] = "HillReporter"
    channel_yt_twitter.loc[161, "Twitter Handle"] = "modern_liberals"
    channel_yt_twitter.loc[247, "Twitter Handle"] = "_UhuruNews_"
    channel_yt_twitter.loc[257, "Twitter Handle"] = "DworkinReport"
    channel_yt_twitter.loc[300, "Twitter Handle"] = "Upworthy"
    channel_yt_twitter.loc[313, "Twitter Handle"] = "Wonkette"

    channel_yt_twitter.loc[836, "Twitter Handle"] = "EverettHerald"
    channel_yt_twitter.loc[1729, "Twitter Handle"] = "DailyO_"
    channel_yt_twitter.loc[2494, "Twitter Handle"] = ""
    channel_yt_twitter.loc[2564, "Twitter Handle"] = "fpmag"
    channel_yt_twitter.loc[2867, "Twitter Handle"] = ""
    channel_yt_twitter.loc[2885, "Twitter Handle"] = ""
    channel_yt_twitter.loc[2887, "Twitter Handle"] = ""

    save_csv_file = "data/2_1. review_twitter_handle.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV saved!")
    print("\n")

if __name__ == '__main__':
    main()