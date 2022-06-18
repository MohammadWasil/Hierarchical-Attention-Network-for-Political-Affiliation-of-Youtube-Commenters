import os, time, json, requests, re, random, bz2, codecs
from datetime import datetime
from pytz import timezone
from html import unescape
from xml.etree import ElementTree
from youtubesearchpython import *

from utils import Timer, search_dict

HTML_TAG_REGEX = re.compile(r'<[^>]*>', re.IGNORECASE)
PT_TIMEZONE = timezone('US/Pacific')

def parse_transcript(plain_data):
    return [{'text': re.sub(HTML_TAG_REGEX, '', unescape(xml_element.text)),
             'start': float(xml_element.attrib['start']),
             'duration': float(xml_element.attrib.get('dur', '0.0'))}
            for xml_element in ElementTree.fromstring(plain_data) if xml_element.text is not None]

def get_video_metadata(video_id):
    timer = Timer()
    timer.start()

    USER_AGENT_LIST = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36',
                   'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36']

    print('>>> now crawling video_id: {0}'.format(video_id))

    num_block = 0
    num_fail = 0
    while True:
        if num_block > 3:
            raise Exception('xxx error, IP blocked, STOP the program...')

        if num_fail > 5:
            print('xxx error, too many fails for video {0}'.format(video_id))
            return {}

        session = requests.Session()
        session.headers['User-Agent'] = random.choice(USER_AGENT_LIST)

        response = session.get("https://www.youtube.com/watch?v={}".format(video_id))
        # too many requests, IP is banned by YouTube
        if response.status_code == 429:
            print('xxx error, too many requests, sleep for 5 minutes, iteration: {0}'.format(num_block))
            num_block += 1
            time.sleep(300)
            continue
        if response is not None:
            html = response.text

            prefix_player = 'window["ytInitialPlayerResponse"] = '
            suffix_player = '\n'
            if prefix_player not in html:
                prefix_player = 'var ytInitialPlayerResponse = '
                suffix_player = ';var'
            prefix_data = 'window["ytInitialData"] = '
            suffix_data = '\n'
            if prefix_data not in html:
                prefix_data = 'var ytInitialData = '
                suffix_data = ';</'

            try:
                #initial_player_response = json.loads(find_value(html, prefix_player, 0, suffix_player).rstrip(';'))
                # print(json.dumps(initial_player_response))
                session_token_begin = response.text.find(prefix_player) + len(prefix_player) + 0
                session_token_end = response.text.find(suffix_player, session_token_begin)
                session_token = response.text[session_token_begin:session_token_end]

                initial_player_response = json.loads(session_token.rstrip(';'))

                #initial_data = json.loads(find_value(html, prefix_data, 0, suffix_data).rstrip(';'))
                # print(json.dumps(initial_data))
                # find response token
                session_token_begin = response.text.find(prefix_data) + len(prefix_data) + 0
                session_token_end = response.text.find(suffix_data, session_token_begin)
                session_token = response.text[session_token_begin:session_token_end]

                initial_data = json.loads(session_token.rstrip(';'))
            except:
                num_fail += 1
                continue
            
            # can also add another way to gather the published date /..................................................................
            if 'videoDetails' not in initial_player_response \
                    or 'microformat' not in initial_player_response \
                    or 'playerMicroformatRenderer' not in initial_player_response['microformat']:
                print('xxx private or unavailable video {0}'.format(video_id))
                return {}

            video_details = initial_player_response['videoDetails']
            microformat_renderer = initial_player_response['microformat']['playerMicroformatRenderer']

            title = video_details['title']
            channel_id = video_details['channelId']
            category = microformat_renderer.get('category', '')
            print('>>> video_id: {0}, category: {1}'.format(video_id, category))

            keywords = video_details.get('keywords', [])
            description = video_details['shortDescription']
            duration = video_details['lengthSeconds']
            publish_date = microformat_renderer['publishDate']
            snapshot_pt_time = datetime.now(PT_TIMEZONE).strftime('%Y-%m-%d-%H')
            view_count = microformat_renderer.get('viewCount', 0)
            is_streamed = video_details['isLiveContent']

            num_like = 0
            num_dislike = 0
            for toggle_button_renderer in search_dict(initial_data, 'toggleButtonRenderer'):
                icon_type = next(search_dict(toggle_button_renderer, 'iconType'), '')
                if icon_type == 'LIKE':
                    default_text = next(search_dict(toggle_button_renderer, 'defaultText'), {})
                    if len(default_text) > 0:
                        try:
                            num_like = int(default_text['accessibility']['accessibilityData']['label'].split()[0].replace(',', ''))
                        except:
                            pass
                elif icon_type == 'DISLIKE':
                    default_text = next(search_dict(toggle_button_renderer, 'defaultText'), {})
                    if len(default_text) > 0:
                        try:
                            num_dislike = int(default_text['accessibility']['accessibilityData']['label'].split()[0].replace(',', ''))
                        except:
                            pass

            ret_json = {'vid': video_id, 'title': title, 'channel_id': channel_id, 'category': category,
                        'keywords': keywords, 'description': description, 'duration': duration,
                        'publish_date': publish_date, 'snapshot_pt_time': snapshot_pt_time, 'is_streamed': is_streamed,
                        'view_count': view_count, 'num_like': num_like, 'num_dislike': num_dislike,
                        'lang': 'NA', 'transcript': 'NA', 'num_comment': 'NA', 'comment_list': 'NA', 'top_comment_list': 'NA'}

            # early return if non-political video
            if category not in ['News & Politics', 'Nonprofits & Activism', 'People & Blogs', 'Education', 'Entertainment', 'Comedy']:
                print('xxx non-political video {0}, category: {1}\n'.format(video_id, category))
                return ret_json
            # print('>>> Step 1: finish getting the metadata via parsing the html page...')

            # 2. get video English transcript
            transcript_response = ''
            caption_tracks = next(search_dict(initial_player_response, 'captionTracks'), [])
            for caption_track in caption_tracks:
                if caption_track['languageCode'] == 'en':
                    if 'kind' in caption_track and caption_track['kind'] == 'asr':
                        transcript_url = caption_track['baseUrl']
                        transcript_response = session.get(transcript_url)
                        break
                    else:
                        transcript_url = caption_track['baseUrl']
                        transcript_response = session.get(transcript_url)
                        break

            if transcript_response != '':
                eng_subtitle = ''
                for segment in parse_transcript(transcript_response.text):
                    if 'text' in segment:
                        seg_text = segment['text'].lower()
                        eng_subtitle += seg_text
                        eng_subtitle += ' '
                ret_json['transcript'] = eng_subtitle
            # print('>>> Step 2: finish getting video English transcript...')

            # 3. get video comments via yt_comments package
            # You can either pass an ID or a URL
            # video_id = "y7zrjWp3fEk-3OmJ0"
            try:
                comments = Comments(video_id)
                
                while comments.hasMoreComments:
                    try:
                        comments.getNextComments()        
                    except:
                        print("Error ... breaking up the code...")
                        comments.hasMoreComments = False
                        break

                print(f'Comments Retrieved: {len(comments.comments["result"])}')
                print('Found all the comments.')

                print("Processing list ... ")
                comment_list = []
                for c in comments.comments["result"]:
                    comment_list.append(create_comment_dict(c))

                comments.comments["result"] = []

                ret_json['num_comment'] = len(comment_list)
                ret_json['top_comment_list'] = comment_list
                #ret_json['comment_list'] = No Idea about this
                
            except Exception as e:
                if 'Comments are turned off' in str(e):
                    print('>>> Comments are turned off')
                else:
                    print('xxx exception in downloading the comments,', str(e))
                pass
            
            # print('>>> Step 3: finish getting video comments...')

            timer.stop()

            time.sleep(5)
            
            return ret_json
    return {}

def create_comment_dict(result):
    author_id = None
    author_name = None
    comment = None
    date_comment = None
    num_replies = None
    
    comment_dict = {}

    if 'author' in result:
        if 'id' in result["author"]:
            author_id = result["author"]["id"]
        if 'name' in result["author"]:
            author_name = result["author"]["name"]

    if 'content' in result:
        comment = result["content"]

    if 'published' in result:
        date_comment = result["published"]

    if 'replyCount' in result:
        num_replies = result["replyCount"]

    comment_dict["author_id"] = author_id
    comment_dict["author_name"] = author_name
    comment_dict["comment"] = comment
    comment_dict["published_date"] = date_comment
    comment_dict["num_of_replies"] = num_replies
    
    return comment_dict

def main():
    
    # complete this ...
    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"
    input_file = os.path.join(DIRECTORY_PATH, "data/6. video_ids.json")

    output_file = "data/7. comments.json"
    num_of_channels_scraped = 0
    visited_channel = set()

    if os.path.exists(os.path.join(DIRECTORY_PATH, output_file)):
        with open(os.path.join(DIRECTORY_PATH, output_file), 'r') as fin:
            for line in fin:
                video_json = json.loads(line.rstrip())
                
                if 'channel id' in video_json:
                    #for vid in video_json["video_ids"]:
                    visited_channel.add(video_json['channel id'])
        num_of_channels_scraped = len(visited_channel)
    
    print('visited {0} Channel in the past, continue...'.format(num_of_channels_scraped))
    print(visited_channel)
    
    with open(os.path.join(DIRECTORY_PATH, output_file), 'a') as fout:
        with open(input_file, 'r') as fin:
            for line in fin:
                
                channel_videos = json.loads(line.rstrip())
                channel_id = channel_videos['channel id']
                
                if channel_id not in visited_channel: 
                    print("Scraping comments from channel with id {} ".format(channel_id))
                    if 'video_ids' in channel_videos:
                        video_id = channel_videos['video_ids']

                        print("Number of videos to scan for comments: {}".format(len(video_id)))
                        if len(video_id) < 200:
                            processed_result = []
                            for i, v_id in enumerate(video_id):
                                print("Video number: {}".format(i))
                                processed_result.append(get_video_metadata(v_id))

                            # write processed_result to disk here
                            fout.write('{}\n'.format(json.dumps({"channel id" : channel_id, "video_ids" : processed_result})))

                            visited_channel.add(channel_id)
                            print("***************************************************************************")
                        else:
                            print("Skipped !!! Too many for now")
                        
    

if __name__ == '__main__':
    main()