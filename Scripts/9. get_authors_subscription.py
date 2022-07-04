import pandas as pd
import json, os
from utils import DIRECTORY_PATH
from googleapiclient.discovery import build

def main():

    LEANING = "RIGHT"
    data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/8. comments {}.csv".format(LEANING)))

    output_file = os.path.join(DIRECTORY_PATH, "data/9. authors_subscription {}.json".format(LEANING))
    
    author_ids = list(set(data["Author Id"].tolist()))[1:]
    num_of_authors_scraped = 0
    visited_authors = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as fin:
            for line in fin:
                #print(line)
                author_subs = list(json.loads(line.rstrip()).keys())[0]
                visited_authors.add(author_subs)
        num_of_authors_scraped = len(visited_authors)
        
    print("Already scraped {} authors".format(num_of_authors_scraped))
    print(visited_authors)

    non_scrapable_auth = os.path.join(DIRECTORY_PATH, "data/9. non_scrapable {}.txt".format(LEANING))
    num_non_scrapable_auth = 0
    visited_non_scrapable_auth = set()
    if os.path.exists(non_scrapable_auth):
        with open(non_scrapable_auth, 'r') as fin:
            for line in fin:
                visited_non_scrapable_auth.add(line)
        num_non_scrapable_auth = len(visited_non_scrapable_auth)

    print("Already visited {} non-scrapable authors".format(num_non_scrapable_auth))
    print(visited_non_scrapable_auth)

    api_key = "AIzaSyA6tGLq3LPKVG3DlRlHK6R5S-yKSCzbPIU"  # Please set your API key
    api_service_name = "youtube"
    api_version = "v3"
    youtube = build(api_service_name, api_version, developerKey=api_key)
    
    with open(non_scrapable_auth, 'a') as fout_non_scrapable:
        with open(output_file, 'a') as fout:
            for auth_id in author_ids:
                if auth_id not in visited_authors:
                    if auth_id not in visited_non_scrapable_auth:

                        request = youtube.subscriptions().list(
                            part="id, snippet",
                            channelId=auth_id,
                            maxResults = 50
                        )
                        try:
                            response = request.execute()
                            authors_subscription = {}
                            channel_subscription = {}
                            for res in response['items']:    
                                channel_id_subscribed   = res['snippet']['resourceId']['channelId']
                                channel_name_subscribed = res['snippet']['title']
                                channel_subscription[channel_id_subscribed] = channel_name_subscribed
                            authors_subscription[auth_id] = channel_subscription
                            fout.write('{}\n'.format(json.dumps(authors_subscription)))
                        except Exception as e:
                            fout_non_scrapable.write('{}\n'.format(auth_id))
                            print(e)
    

if __name__ == '__main__':
    main()