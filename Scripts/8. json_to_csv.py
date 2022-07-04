import pandas as pd
import json, os
from utils import DIRECTORY_PATH

def main():

    # change these three variables accprdingly
    input_file = os.path.join(DIRECTORY_PATH, "data/7. comments RIGHT.json")
    output_file = "data/8. comments RIGHT.csv"
    channel_leaning = "RIGHT"

    channel_id = []
    category = []
    title = []
    video_id = []

    author_id = []
    author_name = []
    comments = []

    number_comments = []

    with open(input_file, 'r') as fin:
        for line in fin:
            channel_videos = json.loads(line.rstrip())
            
            for v_id in channel_videos["video_ids"]:
                #print(v_id)

                for comment in v_id["top_comment_list"]:

                    channel_id.append(v_id['channel_id'])
                    category.append(v_id['category'])
                    title.append(v_id['title'])
                    video_id.append(v_id['vid'])
                    
                    if isinstance(comment, dict):
                        
                        author_id.append(comment["author_id"])
                        author_name.append(comment["author_name"])
                        comments.append(comment["comment"])
                        number_comments.append(v_id["num_comment"])
                        
                    else:
                        author_id.append("NA")
                        author_name.append("NA")
                        comments.append("NA")
                        number_comments.append(0)
                        break

    channel_data = pd.DataFrame(
        {'Biasness': channel_leaning,
         'Channel Id': channel_id,
         'Video Title': title,
         'Video Id': video_id,
         'Video Category': category,
         'Total Comments': number_comments,
         'Author Id': author_id,
         'Author Name': author_name,
         'Authors Comment': comments
        })
    
    # remove rows with no comments
    channel_data = channel_data[channel_data["Total Comments"] != 0]
    channel_data.reset_index(drop=True, inplace=True)

    # save csv
    save_csv_file = os.path.join(DIRECTORY_PATH, output_file)
    channel_data.to_csv(save_csv_file, encoding='utf-8', index=False)
    print("CSV updated!")

if __name__ == '__main__':
    main()