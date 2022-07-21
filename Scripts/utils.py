from urllib.parse import parse_qs, quote, unquote, urlparse
import re
from tweepy import OAuthHandler, API
from googleapiclient.discovery import build
import time
import json
import requests
import scrapetube

import datetime
from bs4 import BeautifulSoup

import time
from datetime import timedelta
import datetime

DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

#left_hashtags = ["LiberalClueless","studentloanslivesmatter","BOYCOTTFRITOLAYS","solidarity","FreeBritney","FREEDMEN","ADOS", "TRUTH", "freepalestine", "PalestinianLivesMatter", "FREEPALESTINIAN", "DefundthePolice", "BLM", "EatTheRich", "CODEPINK", "climatechange", "Whee", "ModiHtaoDeshBchao", "modi_rojgar_do", "Truenews", "victimisation", "freeTheOuighours", "SosColombia", "GazaUnderAttack", "savesheikhjarrah", "BlackLiveMatter", "palestine", "govegan", "Propagande", "pasdamalgame", "videoscopie", "vidÃ©oscopie", "OuÃ¯gours", "Biden"]
#right_hashtags = ["cowards", "generalstrike", "DEBO", "TheFederalist", "TRUMPWON", "trump2024baby", "TuckkkerCarlson", "letsgobrandon", "FJB", "unacceptable", "DemonRats", "BJP", "BOYCOTT", "bantwitter", "dalal", "MatvinScott", "victimisation", "Propagande", "AlexandriaOcasioSmollett", "pasdamalgame", "Impeachbiden", "BLOODYBIDEN"]

number_dictionary = {"1" : "one", "2" : "two", "3": "three", "4" : "four", "5": "five", "6": "six", "7" : "seven", "8": "eight", "9":"nine", "0":"zero"}
dictionary  = {"'cause": 'because',
 "'s": 'is',
 "'tis": 'it is',
 "'twas": 'it was',
 "Ha'ta": 'even',
 "I'd": 'I had',
 "I'll": 'I shall',
 "I'm": 'I am',
 "I'm'a": 'I am about to',
 "I'm'o": 'I am going to',
 "I've": 'I have',
 "S'e": 'oh yeah',
 "ain't": 'am not',
 "aren't": 'are not',
 "cain't": 'cannot',
 "can't": 'cannot',
 "could've": 'could have',
 "couldn't": 'could not',
 "couldn't've": 'could not have',
 "daren't": 'dare not',
 "daresn't": 'dare not',
 "dasn't": 'dare not',
 "didn't": 'did not',
 "doesn't": 'does not',
 "don't": 'do not',
 "e'er": 'ever',
 "everyone's": 'everyone is',
 'finna': 'fixing to',
 'gimme': 'give me',
 "giv'n": 'given',
 "gon't": 'go not',
 'gonna': 'going to',
 'gotta': 'got to',
 "hadn't": 'had not',
 "hasn't": 'is not',
 "haven't": 'have not',
 "he'd": 'he had',
 "he'll": 'he shall',
 "he's": 'he is',
 "he've": 'he have',
 "how'd": 'how did',
 "how'll": 'how will',
 "how're": 'how are',
 "how's": 'how is',
 'howdy': 'how do you do',
 "isn't": 'is not',
 "it'd": 'it would',
 "it'll": 'it shall',
 "it's": 'it is',
 "let's": 'let us',
 "ma'am": 'madam',
 "may've": 'may have',
 "mayn't": 'may not',
 "might've": 'might have',
 "mightn't": 'might not',
 "must've": 'must have',
 "mustn't": 'must not',
 "mustn't've": 'must not have',
 "ne'er": 'never',
 "needn't": 'need not',
 "o'clock": 'of the clock',
 "o'er": 'over',
 "ol'": 'old',
 "oughtn't": 'ought not',
 'rarely': 'cannot',
 "shalln't": 'shall not',
 "shan't": 'shall not',
 "she'd": 'she had',
 "she'll": 'she shall',
 "she's": 'she is',
 "should've": 'should have',
 "shouldn't": 'should not',
 "shouldn't've": 'should not have',
 "so're": 'so are',
 "somebody's": 'somebody is',
 "someone's": 'someone is',
 "something's": 'something is',
 "that'd": 'that would',
 "that'll": 'that shall',
 "that're": 'that are',
 "that's": 'that is',
 "there'd": 'there had',
 "there'll": 'there shall',
 "there're": 'there are',
 "there's": 'there is',
 "these're": 'these are',
 "they'd": 'they had',
 "they'll": 'they shall',
 "they're": 'they are',
 "they've": 'they have',
 "this's": 'this is',
 "those're": 'those are',
 "to've": 'to have',
 "wasn't": 'was not',
 "we'd": 'we had',
 "we'd've": 'we would have',
 "we'll": 'we will',
 "we're": 'we are',
 "we've": 'we have',
 "weren't": 'were not',
 "what'd": 'what did',
 "what'll": 'what shall',
 "what're": 'what are',
 "what's": 'what is',
 "what've": 'what have',
 "when's": 'when is',
 "where'd": 'where did',
 "where're": 'where are',
 "where's": 'where is',
 "where've": 'where have',
 "which's": 'which is',
 "who'd": 'who would',
 "who'd've": 'who would have',
 "who'll": 'who shall',
 "who're": 'who are',
 "who's": 'who is',
 "who've": 'who have',
 "whom'st": 'whom hast',
 "whom'st'd've": 'whom hast had have',
 "why'd": 'why did',
 "why're": 'why are',
 "why's": 'why is',
 "won't": 'will not',
 "would've": 'would have',
 "wouldn't": 'would not',
 "y'all": 'you all',
 "y'all'd've": 'you all would have',
 "you'd": 'you had',
 "you'll": 'you shall',
 "you're": 'you are',
 "you've": 'you have'}



# Review the questions:
def mappingWords(questions,dictionary):
    return " ".join([dictionary.get(w,w) for w in questions.split()])

def preprocess(text, dictionary):

    text = mappingWords(text, dictionary)
    text = text.lower()
    text = mappingWords(text, dictionary)
    text = re.sub(r"[^A-Za-z0-9 ]", "", text)
    return text

def homogeneity_score(r, l):
    # -1 means left ( if < 0)
    # +1 means right ( if > 0)
    return (r-l) / (r+l)

def biasness_classification(homogeneityscore):
    if homogeneityscore > 0:
        return "RIGHT"
    elif homogeneityscore < 0:
        return "LEFT"

def youtube_search_bar(channel_name):
    r = "https://www.youtube.com/results?search_query={}".format(channel_name)
    return r

def recursive_lookup(k, d):
    # imporvised from https://stackoverflow.com/questions/48314755/find-keys-in-nested-dictionary
    # here, k is the key to search
    # d is the dictonary to find from.
    if k in d: return d[k]
    for v in d.values():
        if isinstance(v, dict):
            a = recursive_lookup(k, v)
            if a is not None: yield a
        elif isinstance(v, list):
            for v_list in v:
                if isinstance(v_list, dict):
                    b = recursive_lookup(k, v_list)
                    if b is not None: yield b
    return None

def search_dict(partial, key):
    if isinstance(partial, dict):
        for k, v in partial.items():
            if k == key:
                yield v
            else:
                for o in search_dict(v, key):
                    yield o
    elif isinstance(partial, list):
        for i in partial:
            for o in search_dict(i, key):
                yield o

def check_youtube_for_website_link(soup, website_links, twitter_handle_selected):
    domain = get_domain(website_links)
    
    aid = soup.find('script',string=re.compile('ytInitialData')).text
    yt_about_page = json.loads(aid[20:-1])

    #description = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['description']['simpleText']

    #youtube_to_website_link = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['primaryLinks'][0]['navigationEndpoint']['urlEndpoint']['url']

    #twitter_handle_link = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['primaryLinks'][2]['navigationEndpoint']['urlEndpoint']['url']

    # we will use one of them to compare whether the youtube channel we got indeed belongs to that neews channel or not.
    #links_for_checking_authenticity = [youtube_to_website_link, twitter_handle_link]

    links = next(search_dict(yt_about_page, 'primaryLinks'), [])
    links_for_checking_authenticity = []
    for link in links:
        #pp.pprint(link)
        #print("********************")
        #print(link["navigationEndpoint"]["urlEndpoint"]["url"])
        links_for_checking_authenticity.append(link["navigationEndpoint"]["urlEndpoint"]["url"])

    #website_links = "https://aldianews.com"
    #twitter_handle_selected = "ALDIANews"
    for link in links_for_checking_authenticity:
        #print(link)
        redirected_link = parse_qs(unquote(urlparse(link).query))
        #print(redirected_link)
        if 'q' in redirected_link:
            if "http:" in redirected_link['q'][0]:
                redirected_url = get_domain(redirected_link['q'][0])
                #print(redirected_url)
            else:
                redirected_url = re.sub(r'(https?://)?(www.)?', '', redirected_link['q'][0]).split('/', 1)[0]
                #print(redirected_url)

            if "twitter.com" in redirected_link['q'][0]:
                match = re.search(r"^.*?\btwitter\.com/@?([^?/,\r\n]+)(?:[?/,].*)?$", redirected_link['q'][0])                   
                #print(match)
                #print(match.group(1))
                if match is not None:
                    if twitter_handle_selected == match.group(1):
                        #print(True)
                        return True
            elif redirected_url == domain:
                #print(True)
                return True
    # can add one more check - domain from description.
    # if we find domain in the description of the youtube about page, then also we can say
    # that we ahve matched the correct youtube channel for that news channel website
    description = next(search_dict(yt_about_page, 'description'), {})
    if len(description) > 0:
        #print(description)
        if domain in description["simpleText"]:
            print("yes, the domain is in the description!")
            return True
    return False

def check_twitter_for_website_link(website_domain, twitter_json):
    """
    To check if the twitter handle we are going to select would belongs to the news channel that we selected during that iteartion.
    website_domain : website that we selected from csv file.
    twitter_json : the html file in json format that we extracted using lookup_users() function.
    
    returns : Boolean
                True: if the twitter handle does belongs to the new channel.
                False: if the twitter handle does NOT belongs to the new channel.
    """
    print("checking ... ")
    if "entities" in twitter_json:
        if "url" in twitter_json["entities"]:
            if "urls" in twitter_json["entities"]["url"]:
                if isinstance(twitter_json["entities"]["url"]["urls"], list):
                    if len(twitter_json["entities"]["url"]["urls"]) > 0:
                        if "expanded_url" in twitter_json["entities"]["url"]["urls"][0]:
                            if twitter_json["entities"]["url"]["urls"][0]["expanded_url"] is not None:  
                                expanded_url = twitter_json["entities"]["url"]["urls"][0]["expanded_url"]
                                if get_domain(website_domain) == get_domain(expanded_url):
                                    if urlparse(get_domain(website_domain)).path == "" or urlparse(get_domain(website_domain)).path == "/":
                                        if urlparse(get_domain(expanded_url)).path == "" or urlparse(get_domain(expanded_url)).path == "/":
                                            print("True")
                                            return True
                                    elif urlparse(get_domain(website_domain)).path.strip("/") == urlparse(get_domain(expanded_url)).path.strip("/"):
                                        print("True")
                                        return True
    print("False")
    return False

def get_domain(url):
    """
    To get th domain form a link
    For eg. http://www.facebook.com -> get_domain("http://www.facebook.com") -> facebook.com
    From github:
    
    """
    return re.sub(r'www\d?.', '', urlparse(url.lower()).netloc.split(':')[0])

def Authenticate_twitter(cfg):
    # Authenticate to Twitter
    auth = OAuthHandler(cfg["API_KEYS"], cfg["API_SECRET_KEYS"])
    auth.set_access_token(cfg["ACCESS_TOKEN"], cfg["ACCESS_TOKEN_SECRET"])

    api = API(auth, wait_on_rate_limit=True)

    if api.verify_credentials() == False:
        print("The user credentials are invalid.")
    else:
        print("The user credentials are valid.")

    return api

def Authenticate_youtube(cfg):
    youtube_service = build('youtube', 'v3', developerKey=cfg["YOUTUBE_KEY"])
    return youtube_service

def get_youtube_channel_id(service, part_id, youtube_user):
    """
    Get youtube channel id with the given youtube user.
    returns : youtube channel id.
    """
    for i in range(3):
        try:
            response = service.channels().list(part = part_id, forUsername=youtube_user)
            response = response.execute()
            #print(response)
            # if there is a response
            if response is not None:
                if "items" in response:
                    if isinstance(response["items"], list):
                        if len(response["items"][0]) > 0:
                            #print("yes")
                            #print(response["items"][0]["id"])
                            channel_id = response["items"][0]["id"]
                            return channel_id
            else:
                return ""

        except:
            time.sleep(2**i)
            print("exception!")
    return ""

def check_youtube_channel_id(service, part_id, youtube_id):
    """
    Get youtube channel id with the given youtube user.
    """
    for i in range(3):
        try:
            response = service.channels().list(part = part_id, id=youtube_id)
            response = response.execute()
            #print(response)
            # if there is a response
            if response is not None:
                if "items" in response:
                    if isinstance(response["items"], list):
                        if len(response["items"][0]) > 0:
                            #print("yes")
                            #print(response["items"][0]["id"])
                            channel_id = response["items"][0]["id"]
                            return channel_id
            else:
                return ""
        except:
            time.sleep(2**i)
            print("exception!")
    return ""

def ajax_request(session, url, params=None, data=None, headers=None, max_retry=5, sleep=20):
    for idx_request in range(max_retry):
        response = session.post(url, params=params, data=data, headers=headers)
        time.sleep(1)
        if response.status_code == 200:
            ret_json = response.json()
            if isinstance(ret_json, list):
                ret_json = [x for x in ret_json if 'response' in x][0]
            ret_json.update({'num_request': idx_request + 1})
            return ret_json
        elif response.status_code in [403, 413]:
            return {'error': 'Comments are turned off', 'num_request': idx_request + 1}
        time.sleep(sleep)
    return {'error': 'Unknown error {0}'.format(response.status_code), 'num_request': max_retry}

def get_video_date_upload(video_id):
    youtube_video_link = "https://www.youtube.com/watch?v={}".format(video_id)

    session = requests.Session()
    header = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}

    response = session.get(youtube_video_link, headers=header)

    if response.status_code == 429:
        print("Blocked by Youtube!")
        return None
    if response is not None:
        soup = BeautifulSoup(response.text, features='html.parser')
        try:
            aid = soup.find('script',string=re.compile('ytInitialData')).text
            aid = json.loads(aid[20:-1])
        except:
            return None

        if 'videoDetails' not in aid or 'microformat' not in aid or 'playerMicroformatRenderer' not in aid['microformat']:
            #print('xxx private or unavailable video {0}'.format(video_id))

            # trying manual scraping
            for date in search_dict(aid, 'dateText'):
                try:
                    date = datetime.datetime.strptime(date["simpleText"], "%d.%m.%Y").strftime("%Y-%m-%d")
                    return datetime.datetime.strptime(date, "%Y-%m-%d")
                except:
                    return None
        else:
            microformat_renderer = aid['microformat']['playerMicroformatRenderer']
            publish_date = microformat_renderer['publishDate']
            return publish_date

def get_video_ids_playlist(channel_id):

    START_DATE = datetime.datetime.strptime('2021-01-01', "%Y-%m-%d")
    END_DATE = datetime.datetime.strptime('2021-08-31', "%Y-%m-%d")

    channel_get_all_videos_playlist = "UU" + channel_id[2:]

    playlist = scrapetube.get_playlist(channel_get_all_videos_playlist)

    playlist_videos = []
    selected_videos = []

    for _, vid in enumerate(playlist):
        #print(i, vid['videoId'])
        playlist_videos.append(vid['videoId'])

    print("There are {} number of youtube videos in channel id {}: ".format(len(playlist_videos), channel_id))
        
    if len(playlist_videos) > 0:
        for video in playlist_videos:
            earliest_publish_date = get_video_date_upload(video)
            if earliest_publish_date is not None:
                if earliest_publish_date > START_DATE :
                    if  earliest_publish_date < END_DATE:
                        print("Date: ", earliest_publish_date, " | Video Id : ", video)
                        selected_videos.append(video)
                else:
                    # since all the videos are in order, we do not want the rest of the comments form the videos
                    #print("Dont ADD")
                    return selected_videos # remove break and add return statement
            else:
                #print("Received Unknown type of Date: Passing ... ")
                pass
    return selected_videos # return statement

def get_video_ids_playlist2(channel_get_all_videos_playlist):

    START_DATE = datetime.datetime.strptime('2021-01-01', "%Y-%m-%d")

    session = requests.Session()
    header = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}
    response = session.get("https://www.youtube.com/playlist?list={}".format(channel_get_all_videos_playlist), headers=header)

    if response is not None:
        # find response token
        session_token_begin = response.text.find("XSRF_TOKEN") + len("XSRF_TOKEN") + 3
        session_token_end = response.text.find('"', session_token_begin)
        session_token = response.text[session_token_begin:session_token_end]

        soup = BeautifulSoup(response.text, features='html.parser')
        aid = soup.find('script',string=re.compile('ytInitialData')).text
        aid = json.loads(aid[20:-1])
        playlist_videos = []
        selected_videos = []
        
        for video in search_dict(aid, 'playlistVideoRenderer'):
            playlist_videos.append(video['videoId'])
        
        if len(playlist_videos) > 0:
            for video in playlist_videos:
                earliest_publish_date = get_video_date_upload(video)
                if earliest_publish_date is not None:
                    print("date: ", earliest_publish_date)
                    if earliest_publish_date > START_DATE:
                        print("ADD: ", video) # return
                        selected_videos.append(video)
                    else:
                        # since all the videos are in order, we do not want the rest of the comments form the videos
                        print("Dont ADD")
                        return selected_videos # remove break and add return statement
                        #break
                else:
                    print("Received Unknown type of Date: Passing ... ")
        
        # empty this list, so as to add new videos from playlist
        playlist_videos = []
        ncd = next(search_dict(aid, 'continuationEndpoint'), None)
        if ncd:
            print("Continuation exists !!!")
            continuations = [(ncd['continuationCommand']['token'], ncd['clickTrackingParams'])]

            while continuations:
                continuation, itct = continuations.pop()
                response_json = ajax_request(session,
                                            'https://www.youtube.com/browse_ajax',
                                            params={'referer': "https://www.youtube.com/playlist?list={}".format(channel_get_all_videos_playlist),
                                                    'pbj': 1,
                                                    'ctoken': continuation,
                                                    'continuation': continuation,
                                                    'itct': itct},
                                            data={'session_token': session_token},
                                            headers={'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
                                                    'X-YouTube-Client-Name': '1',
                                                    'X-YouTube-Client-Version': '2.20200207.03.01'})

                if len(response_json) == 0:
                    break
                #if next(search_dict(response_json, 'externalErrorMessage'), None):
                #    raise Exception('Error returned from server: ' + next(search_dict(response_json, 'externalErrorMessage')))
                #elif 'error' in response_json:
                #    raise Exception(response_json['error'])

                # Ordering matters. The newest continuations should go first.
                continuations = [(ncd['continuationCommand']['token'], ncd['clickTrackingParams']) for ncd in search_dict(response_json, 'continuationEndpoint')] + continuations

                for video in search_dict(response_json, 'playlistVideoRenderer'):
                    playlist_videos.append(video['videoId'])

                if len(playlist_videos) > 0:
                    for video in playlist_videos:
                        earliest_publish_date = get_video_date_upload(video)
                        if earliest_publish_date is not None:
                            print("date: ", earliest_publish_date)
                            if earliest_publish_date > START_DATE:
                                print("ADD: ", video) # return
                                selected_videos.append(video)
                            else:
                                # since all the videos are in order, we do not want the rest of the comments form the videos
                                print("Dont ADD")
                                return selected_videos # remove break and add return statement
                        else:
                            print("Received Unknown type of Date: Passing ... ")
    return selected_videos

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print('>>> Elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - self.start_time))[:-3]))


def strify(lst, delim=','):
    return delim.join(map(str, lst))


def intify(lst_str, delim=','):
    return list(map(int, lst_str.split(delim)))


def bias_metric(a, b):
    return (b - a) / (a + b)
