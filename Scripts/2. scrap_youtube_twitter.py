import time, requests, re
import os
from bs4 import BeautifulSoup
from collections import defaultdict

import pandas as pd
import pprint
import json
import yaml
from difflib import SequenceMatcher
import time, requests, re, random, tldextract

from utils import check_twitter_for_website_link, check_youtube_for_website_link, get_domain, Authenticate_twitter

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

    # load config_KEYS.yaml file
    with open(os.path.join(DIRECTORY_PATH, "utility/config_KEYS.yml"), "r") as ymlfile:
        cfg =  yaml.safe_load(ymlfile)

    start = time.time()
    api = Authenticate_twitter(cfg)

    file_path = "data/News channels.csv"
    news_channels = pd.read_csv(os.path.join(DIRECTORY_PATH, file_path))

    num_channels = 0
    num_twitter_handle = 0
    num_youtube_channels = 0
    num_fail_attempts = 0
    num_twitter_handle_to_review = 0

    twitter_handle_sim_score = defaultdict(float)
    new_channel_yt_twitter_sim = pd.DataFrame(columns=['Name', 'Bias Rating', 'Country', 'Website', 'Youtube Channel', 'Twitter Handle', 'Twitter Similarity'])

    sites_to_ignore = ["https://www.bellinghamherald.com/", "https://www.charlotteobserver.com", "https://www.fresnobee.com/",
                        "https://www.kansascity.com", "http://www.kentucky.com/", "http://www.mcclatchydc.com/", "https://www.miamiherald.com",
                        "https://www.newsobserver.com/", "https://www.heraldonline.com/", "https://www.sacbee.com", "https://www.sanluisobispo.com/",
                        "https://www.stuff.co.nz/", "https://www.thenewstribune.com/", "https://www.theolympian.com/", "https://www.thestate.com/",
                        "https://www.usnews.com/", "https://www.bnd.com/", "http://www.idahostatesman.com", "https://www.heraldsun.com", "https://www.macon.com",
                        "https://www.modbee.com/", "https://www.tri-cityherald.com/", "https://www.kansas.com/", "https://www.star-telegram.com/",
                        "https://www.itv.com/news", "https://www.telegraph.co.uk", "https://www.newsmax.com/"]

    for _, website_links in enumerate(news_channels["Website"]):

        # skip this one
        if (website_links not in sites_to_ignore):

            num_channels += 1
            try:
                index = news_channels.index[news_channels["Website"] == website_links].tolist()[0]

                new_channel_yt_twitter_sim.loc[index, "Name"] = news_channels["Name"][index]
                new_channel_yt_twitter_sim.loc[index, "Bias Rating"] = news_channels["Bias Rating"][index]
                new_channel_yt_twitter_sim.loc[index, "Country"] = news_channels["Country"][index]
                new_channel_yt_twitter_sim.loc[index, "Website"] = news_channels["Website"][index]

                twitter_handle = defaultdict(int)
                youtube_channel_id = defaultdict(int)
                youtube_channel_users = defaultdict(int)
                print("{} : {}".format(index, website_links))

                try:
                    response = requests.get(website_links, headers = headers, allow_redirects=False)
                except:
                    response = requests.get(website_links, allow_redirects=False) 
                time.sleep(1)
                if response.text.strip() == '':
                    try:
                        response = requests.get(website_links,
                                                headers={random.choice(USER_AGENT_LIST)},
                                                allow_redirects=True)
                    except:
                        response = requests.get(website_links, allow_redirects=True)

                soup = BeautifulSoup(response.text, 'lxml')

                # trying to find social media link, iei either youtube link or twitter handle.
                website_link = soup.find_all('a', href=True)

                for socialmedia_link in website_link:

                    channel_link = socialmedia_link.get("href")

                    if "twitter.com" in channel_link.lower():
                        if (bool(re.match(r'https://twitter.com/[\w@]+\??|/?$', channel_link)) and not bool(re.match(r'https://twitter.com/(intent|share|home|hashtag|search)/?', channel_link))
                            and not bool(re.match(r'https://twitter.com/[\w]+/status?', channel_link))):

                            # extract the twitter handle
                            #match = re.search(r'^.*?\btwitter\.com/@?(\w{1,15})(?:[?/,].*)?$', socialmedia_link.get("href").lower())
                            match = re.search(r"^.*?\btwitter\.com/@?([^?/,\r\n]+)(?:[?/,].*)?$", channel_link)                   
                            # increament the dictionary with that key as twiter handle
                            twitter_handle[match.group(1)] += 1
                            print("Twitter handle Dictionary: ", twitter_handle)

                    if "youtube.com/" in channel_link:

                        if (not(bool(re.match(r'https://www.youtube.com/watch?[\w]+/?', channel_link)))):

                            # it can be channel id or youtube user
                            # Example: 
                            # https://www. youtube.com/channel/UCUZHFZ9jIKrLroW8LcyJEQQ
                            # https://www.youtube.com/channel/UC_WSavcwZVE3ciFM8yCwkBw/
                            # https://www.youtube.com/channel/UCHdWMuH-IIveBWjIsOhK-dw/videos
                            # https://www.youtube.com/channel/UCXu7fg-_KdAoHY7bmahr9vg/featured
                            # https://www.youtube.com/channel/UCmX0gITcFMtmDSah8-wCChw/videos?view_as=subscriber    
                            if (bool(re.match(r'https://www.youtube.com/channel/[\w]+/?', channel_link))):

                                youtube_id = yt_id = re.split(r'/|\?', channel_link)[4]
                                youtube_id = re.split('[^a-zA-Z0-9_-]', youtube_id)[0]
                                print("Youtube Id", youtube_id)
                                youtube_channel_id[youtube_id] += 1

                            # Example: 
                            # https://www.youtube.com/c/AmericanBridgePAC
                            # https://www.youtube.com/c/cosmopolitan?sub_confirmation=1
                            elif (bool(re.match(r'https://www.youtube.com/c/[\w]+/?', channel_link))):
                                youtube_user = re.split(r'/|\?', channel_link)[4]
                                youtube_user = re.split('\W', youtube_user)[0]
                                print("Youtube User", youtube_user)
                                youtube_channel_users[youtube_user] += 1

                            # Example: 
                            # https://www.youtube.com/user/pontealdiatv
                            # https://www.youtube.com/user/blackamericaweb?sub_confirmation=1
                            elif(bool(re.match(r'https://www.youtube.com/user/[\w]+/?', channel_link))):
                                youtube_user = re.split(r'/|\?', channel_link)[4]
                                youtube_user = re.split('\W', youtube_user)[0]
                                print("Youtube User", youtube_user)
                                youtube_channel_users[youtube_user] += 1

                            # Example: https://www.youtube.com/actdottv
                            elif (bool(re.match(r'https://www.youtube.com/[\w]+/?', channel_link))):
                                youtube_user = re.split(r'/|\?', channel_link)[3]
                                youtube_user = re.split('\W', youtube_user)[0]
                                print("Youtube User", youtube_user)
                                youtube_channel_users[youtube_user] += 1

                if(len(twitter_handle)) > 1:
                    twitter_handle_list = sorted(list(twitter_handle.keys()), key=lambda x: twitter_handle[x], reverse=True)

                    # lookup the user
                    try:
                        returned_users = api.lookup_users(screen_name=twitter_handle_list)
                        for user in returned_users:

                            user_json = user._json
                            screen_name = user_json['screen_name'].lower()
                            print("Screen name: {} / website link: {}  ".format(screen_name, website_links))
                            if check_twitter_for_website_link(website_links, user_json):
                                selected_twitter_handle = screen_name
                                print("Selected twitter handle name: ", selected_twitter_handle)
                                break

                    except:
                        for twitter_handle in twitter_handle_list:
                            returned_user = api.lookup_users(screen_name = [twitter_handle])[0]
                            user_json = returned_user._json
                            screen_name = user_json['screen_name'].lower()

                            print("Screen name: {} / website link: {}  ".format(screen_name, website_links))           
                            if check_twitter_for_website_link(user_json, website_links):
                                selected_twitter_handle = screen_name
                                print("Selected twitter handle name: ", selected_twitter_handle)
                                break

                elif len(twitter_handle)==1:
                    selected_twitter_handle = list(twitter_handle.keys())[0].lower()
                    print("Twitter with only on handle: ", selected_twitter_handle)

                if selected_twitter_handle != '':
                    num_twitter_handle += 1
                    print("website domain name: ", tldextract.extract(get_domain(website_links)).domain)
                    tw_similarity = SequenceMatcher(None, tldextract.extract(get_domain(website_links)).domain, selected_twitter_handle).ratio()

                    new_channel_yt_twitter_sim.loc[index, "Twitter Handle"] = selected_twitter_handle
                    new_channel_yt_twitter_sim.loc[index, "Twitter Similarity"] = tw_similarity
                    
                    # twitter handle to be reviewed
                    twitter_handle_sim_score[selected_twitter_handle] = tw_similarity
                    
                    print("similarity: ", tw_similarity)
                    selected_twitter_handle = ""

                # if we extracted more than one youtube id for a single new channel, we
                # need to select from one of them
                print("Youtube channel Id Dictionary", youtube_channel_id)
                headers= {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}

                select_youtube_id = ""
                if(len(youtube_channel_id) > 1):
                    youtube_id_list = sorted(list(youtube_channel_id.keys()), key=lambda x: youtube_channel_id[x], reverse=True)
                    print(youtube_id_list)
                    for ids in youtube_id_list:
                        print("Youtube Channel link : https://www.youtube.com/channel/{}/about".format(ids))
                        yt_about_page = requests.get("https://www.youtube.com/channel/{}/about".format(ids), headers=headers)
                        time.sleep(1)
                        soup = BeautifulSoup(yt_about_page.content, features='html.parser')

                        if check_youtube_for_website_link(soup, website_links, selected_twitter_handle):
                            select_youtube_id = ids
                            break
                elif len(youtube_channel_id) == 1:
                    select_youtube_id = list(youtube_channel_id.keys())[0]
                print("selected youtube id: ", select_youtube_id)

                if select_youtube_id != "":
                    new_channel_yt_twitter_sim.loc[index, "Youtube Channel"] = select_youtube_id
                
                # Test the above code and write code for select_youtube_user
                print("Youtube channel User Dictionary", youtube_channel_users)

                select_youtube_user = ""
                if (len(youtube_channel_users) > 1):
                    youtube_user_list = sorted(list(youtube_channel_users.keys()), key=lambda x: youtube_channel_users[x], reverse=True)
                    print(youtube_user_list)
                    for users in youtube_user_list:
                        print("Youtube Channel link : https://www.youtube.com/channel/{}/about".format(users))
                        yt_about_page = requests.get("https://www.youtube.com/channel/{}/about".format(users), headers=headers)
                        time.sleep(1)
                        soup = BeautifulSoup(yt_about_page.content, features='html.parser')

                        if check_youtube_for_website_link(soup, website_links, selected_twitter_handle):
                            select_youtube_user = users
                            break
                elif len(youtube_channel_users) == 1:
                    select_youtube_user = list(youtube_channel_users.keys())[0]
                print("selected youtube user: ", select_youtube_user)

                if select_youtube_user != "":
                    new_channel_yt_twitter_sim.loc[index, "Youtube Channel"] = select_youtube_user
                
                # statistics:
                if select_youtube_id != "" or select_youtube_user != "":
                    num_youtube_channels += 1
                print("**************************************************************************************************************")
            except:
                num_fail_attempts += 1
                continue
    print("\n")
    print("**************************************************************************************************************")
    for selected_twitter_handle, tw_similarity in twitter_handle_sim_score.items():           
        if isinstance(tw_similarity, float) and tw_similarity < 0.5:
            num_twitter_handle_to_review += 1
            print("Twitter handles to be reviewed: {} with Similarity: {}".format(selected_twitter_handle, tw_similarity))
    print("**************************************************************************************************************")
    print("\n")
    # save twitter_handle_sim_score as json file
    # whichever twitter ahndle has less than 0.5 similarity, we need to manually check them.
    save_twitter_handle = "data/2. twitter_handle_sim_score.json"
    with open(os.path.join(DIRECTORY_PATH, save_twitter_handle), 'w') as thss:
        json.dump(twitter_handle_sim_score, thss)

    print("**************************************************************************************************************")
    print("************************************************Statistics****************************************************")
    print("**************************************************************************************************************")
    print("Number of twitter handle found/Number of total News channels : {} / {}".format(num_twitter_handle, num_channels))
    print("Number of youtube user/id found/Number of total News channels : {} / {}".format(num_youtube_channels, num_channels))
    print("Number of failed attempts/Total attempts or News channels : {} / {}".format(num_fail_attempts, num_channels))
    print("Number of twitter to be reviewed: {}".format(num_twitter_handle_to_review))
    print("**************************************************************************************************************")
    print("************************************************Statistics****************************************************")
    print("**************************************************************************************************************")
    print("\n")

    save_csv_file = "data/2. scrap_youtube_twitter.csv"
    new_channel_yt_twitter_sim.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV saved!")
    print("\n")
    print('It took {0:0.1f} seconds'.format(time.time() - start))

if __name__ == '__main__':
    main()