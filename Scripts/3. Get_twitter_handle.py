import pandas as pd
import os, yaml, tldextract
from difflib import SequenceMatcher
from utils import check_twitter_for_website_link, get_domain, Authenticate_twitter

DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

def main():

    # load config_KEYS.yaml file
    with open(os.path.join(DIRECTORY_PATH, "utility/config_KEYS.yml"), "r") as ymlfile:
        cfg =  yaml.safe_load(ymlfile)

    api = Authenticate_twitter(cfg)

    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/2. scrap_youtube_twitter.csv"))
    channel_yt_twitter = channel_yt_twitter.reset_index()

    update_csv = False

    # statistics
    num_requests_sent = 0
    num_updates = 0
    num_fail_attempts = 0

    for index, row in channel_yt_twitter.iterrows():

        if pd.isna(channel_yt_twitter.iloc[index, 6]):
            channel_name = channel_yt_twitter.iloc[index, 1]
            website_links = channel_yt_twitter.iloc[index, 4]
            
            returned_users = api.search_users(channel_name, count = 10)
            num_requests_sent += 1
            update_csv = True
            
            for user in returned_users:
                user_json = user._json
                screen_name = user_json['screen_name'].lower()
                print("Screen name: {} / website link: {}  ".format(screen_name, website_links))
                
                if not pd.isna(website_links):
                    if check_twitter_for_website_link(website_links, user_json):

                        selected_twitter_handle = screen_name
                        print("Selected twitter handle name: ", selected_twitter_handle)

                        tw_similarity = SequenceMatcher(None, tldextract.extract(get_domain(website_links)).domain, selected_twitter_handle).ratio()
                        print(tw_similarity)

                        channel_yt_twitter.iloc[index, 6] = selected_twitter_handle
                        channel_yt_twitter.iloc[index, 7] = tw_similarity
                        num_updates += 1
                        update_csv = False
                        break
            if update_csv == True:
                num_fail_attempts += 1
                update_csv = False
                
    print("Number of successfull update {} / Number of requests sent: {}".format(num_updates, num_requests_sent))
    print("Number of Failed update {} / Number of requests sent: {}".format(num_fail_attempts, num_requests_sent))

    save_csv_file = "data/3. scrap_youtube_twitter_handle.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV saved!")

    
if __name__ == '__main__':
    main()