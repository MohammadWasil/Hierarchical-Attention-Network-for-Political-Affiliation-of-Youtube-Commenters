from urllib.parse import parse_qs, quote, unquote, urlparse
import re

import json



def check_youtube_for_website_link(soup, website_links, twitter_handle_selected):
    domain = get_domain(website_links)
    
    aid = soup.find('script',string=re.compile('ytInitialData')).text
    yt_about_page = json.loads(aid[20:-1])

    description = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['description']['simpleText']

    youtube_to_website_link = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['primaryLinks'][0]['navigationEndpoint']['urlEndpoint']['url']

    twitter_handle_link = yt_about_page['contents']['twoColumnBrowseResultsRenderer']['tabs'][5]['tabRenderer']['content']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][0]['channelAboutFullMetadataRenderer']['primaryLinks'][2]['navigationEndpoint']['urlEndpoint']['url']

    # we will use one of them to compare whether the youtube channel we got indeed belongs to that neews channel or not.
    links_for_checking_authenticity = [youtube_to_website_link, twitter_handle_link]

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
                #print(match.group(1))
                if twitter_handle_selected == match.group(1):
                    #print(True)
                    return True
            elif redirected_url == domain:
                #print(True)
                return True
    # can add one more check - domain from description.
    # if we find domain in the description of the youtube about page, then also we can say
    # that we ahve matched the correct youtube channel for that news channel website
    if len(description) > 0:
        #print(description)
        if domain in description:
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