from urllib.parse import parse_qs, quote, unquote, urlparse
import re
from tweepy import OAuthHandler, API
from googleapiclient.discovery import build
import time
import json

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