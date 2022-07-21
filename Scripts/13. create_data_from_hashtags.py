import pandas as pd
import os
from utils import DIRECTORY_PATH, homogeneity_score, biasness_classification
from hashtags import right_hashtags, left_hashtags

def main():

    LEANING = "RIGHT"

    left_hashtags_ = []
    for lh in left_hashtags:
        left_hashtags_.append(lh.lower() )
    
    right_hashtags_ = []
    for rh in right_hashtags:
        right_hashtags_.append(rh.lower() )
    
    print("Reading data ...")
    channel = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/10. comments {}.csv".format(LEANING) ))

    # for the comments from left channels
    authors_who_used_hashtags = []
    for _, row in channel.iterrows():
        if pd.isna(row["Authors Biasness"]):
            if not pd.isna(row["Authors Comment"]):
                for i in row["Authors Comment"].split():
                    if i.startswith("#"):
                        authors_who_used_hashtags.append(row["Author Id"])

    authors_who_used_hashtags = set(authors_who_used_hashtags)

    print("Calculating Homogeneity Score ...")
    users_leaning = {}
    for user_id in authors_who_used_hashtags:
        user_comment = channel[channel["Author Id"] == user_id]["Authors Comment"]
        
        # create one big string of all commmentsform a user.
        #comment = ""
        #for i in user_comment:
        #    comment = comment + i + " "
        
        # calculate the number of left or right hashtags used
        left_user = 0
        right_user = 0

        #for com in comment.split(" "):
        #    if com.replace("#", "").lower() in left_hashtags_:
        #        left_user += 1
        #    elif com.replace("#", "").lower() in right_hashtags_:            
        #        right_user += 1

        for com in user_comment:
            for c in com.split(" "):
                if c.replace("#", "").lower() in left_hashtags_:
                    left_user += 1
                elif c.replace("#", "").lower() in right_hashtags_:            
                    right_user += 1

        if right_user != 0 or left_user != 0:
            homo_score = homogeneity_score(right_user, left_user)
            users_leaning[user_id] = homo_score

    # updating the dataset  
    for key, value in users_leaning.items():
        indx = channel[channel["Author Id"] == key].index

        channel.loc[indx, "Homogeneity Score"] = value
        channel.loc[indx, "Authors Biasness"] = biasness_classification(value)

    save_csv_file = "data/13. comments {} with Hashtag annotations.csv".format(LEANING)
    channel.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("{} CSV updated!".format(channel))


if __name__ == '__main__':
    main()