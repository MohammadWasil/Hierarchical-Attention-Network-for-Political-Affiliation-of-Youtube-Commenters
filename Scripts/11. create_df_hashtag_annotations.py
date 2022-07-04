import pandas as pd
import os

DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"

def main():
    # have to change this
    LEANING = "LEFT"
    channels = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/10. comments {}.csv".format(LEANING)))

    print("CSV Loaded ...")
    print("Finding Hashtags ...")
    author_id = []
    author_comment = []
    hashtags_list = []
    for _, row in channels.iterrows():
        
        # If the users/author of the comment has not been annotated yet as liberal or conservative
        if pd.isna(row["Authors Biasness"]):
                    
            # populate the hashtag_list
            hashtag = []
            for i in row["Authors Comment"].split():
                if i.startswith("#"):
                    hashtag.append(i)
            
            # if there is any hashtags in the comment, then only save it.
            if len(hashtag) > 0:
                author_id.append(row["Author Id"])
                author_comment.append(row["Authors Comment"])
                hashtags_list.append(hashtag)

    channel_hashtags = pd.DataFrame(
            {"Author Id" : author_id,
            'Authors Comment': author_comment,
            'Hashtags': hashtags_list
            })

    save_csv_file = "data/11. Hashtags {}.csv".format(LEANING)
    channel_hashtags.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")

if __name__ == '__main__':
    main()