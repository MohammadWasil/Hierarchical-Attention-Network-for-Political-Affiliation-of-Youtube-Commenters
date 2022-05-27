"""
Scrap the youtube search page using channel title to find their youtube channel.

Input : 3. scrap_youtube_twitter_handle.csv
Output : 4. scrap_youtube_channel.csv
"""

import os, json, time, requests, re, random
import pandas as pd
from bs4 import BeautifulSoup

from utils import youtube_search_bar, check_youtube_for_website_link

def main():
    USER_AGENT_LIST = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36']

    headers = requests.utils.default_headers()
    headers.update(
        {
            'User-Agent': random.choice(USER_AGENT_LIST),
        }
    )

    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/3. scrap_youtube_twitter_handle.csv"))
    channel_yt_twitter.drop("index",axis=1,inplace=True)

    # statistics
    num_youtube_channels_nan = 0
    num_youtube_channels_updates = 0
    num_youtube_channels_fail_attempts = 0

    for index, row in channel_yt_twitter.iterrows():
    
        num_search_results = 0
        
        # if we dont have channel name or its user id
        if pd.isna(channel_yt_twitter.iloc[index, 4]):
            print("*********************")
            print("Channel name: ", channel_yt_twitter.iloc[index, 0])

            num_youtube_channels_nan += 1

            for _ in range(5):
                if num_search_results == 0:
                    # can remove brackets from the name of the channel
                    result_link = youtube_search_bar(channel_yt_twitter.iloc[index, 0])
                    try:
                        response = requests.get(result_link, headers = headers)
                    except:
                        response = requests.get(result_link)
                    time.sleep(1)

                    if response.text:
                        soup = BeautifulSoup(response.text, 'lxml')

                        aid = soup.find('script',string=re.compile('ytInitialData')).text
                        yt_about_page = json.loads(aid[20:-1])
                        
                        #pp.pprint(yt_about_page)

                        result_lists = yt_about_page["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"] \
                        ["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]

                        #result_list = recursive_lookup("channelRenderer", yt_about_page)
                        list_of_retrieved_channel_json = []
                        for r in result_lists:
                            
                            if list(r.keys())[0] == "channelRenderer":
                                print("channel result found on the seach result")
                                list_of_retrieved_channel_json.append(r)
                                num_search_results = len(list_of_retrieved_channel_json)
            
            
            for channel_json in list_of_retrieved_channel_json:    
                channel_id = channel_json["channelRenderer"]["channelId"]
                channel_title = channel_json["channelRenderer"]["title"]["simpleText"]
                print(channel_id, channel_title)
                
                # check if we got the right youtube channel or not.
                # we can compare it with the domain of the website and twitter handle.
                header = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
                if channel_title != "":
                    about_page = "https://www.youtube.com/channel/{}/about".format(channel_id)
                    yt_about_page = requests.get(about_page, headers=header)
                    time.sleep(1)
                    soup = BeautifulSoup(yt_about_page.content, features='html.parser')
                    
                    if check_youtube_for_website_link(soup, channel_yt_twitter.iloc[index, 3], channel_yt_twitter.iloc[index, 5]):
                        print("match found!")
                        # update the csv file.
                        num_youtube_channels_updates += 1
                        channel_yt_twitter.iloc[index, 4] = channel_title
                        # there should be a break here.
                        break
                    else:
                        num_youtube_channels_fail_attempts += 1

    print("Number of successfull youtube channel name scrap update {} / Number of requests sent for youtube channel name scrap: {}".format(num_youtube_channels_updates, num_youtube_channels_nan))
    print("Number of Failed youtube channel name scrap {} / Number of requests sent for youtube channel name scrap: {}".format(num_youtube_channels_fail_attempts, num_youtube_channels_nan))

    save_csv_file = "data/4. scrap_youtube_channel.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV saved!")

if __name__ == '__main__':
    main()