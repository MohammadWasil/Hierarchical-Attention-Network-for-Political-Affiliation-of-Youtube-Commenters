"""
First, try to find out if there are conflicted users (user both in left and right channels with different lenaing), if there 
are, remove them. Then combine both the dataet and save.
Next, take those samples, where we know the leaning of the user, and generate training data for training.
"""


import pandas as pd
import os
from utils import DIRECTORY_PATH


def main():
    print("Reading Data")
    data1 = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/13. comments LEFT with Hashtag annotations.csv"), lineterminator='\n')
    data2 = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/13. comments RIGHT with Hashtag annotations.csv"))

    # get all non null rows of column Authors Biasness
    #dataL = data1[data1["Authors Biasness"].notnull()]
    #dataR = data2[data2["Authors Biasness"].notnull()]

    left_channel_nn = data1[data1["Authors Biasness"].notnull()]
    right_channel_nn = data2[data2["Authors Biasness"].notnull()]

    # list of authors from let nd roght leaning channels, with their biassness (annotated data)
    users_from_left = list(left_channel_nn["Author Id"].unique())
    users_from_right = list(right_channel_nn["Author Id"].unique())

    # check of the author is present on both or not.
    #left_authors = dataL["Author Id"].unique().tolist()
    #right_authors = dataR["Author Id"].unique().tolist()

    """left_in_right = []
    for la in left_authors:
        if la in right_authors:
            left_in_right.append(la)# += 1 # 78

    ids_to_discard = []
    for id_ in left_in_right:
        L = set(dataL[dataL["Author Id"] == id_]["Authors Biasness"])
        R = set(dataR[dataR["Author Id"] == id_]["Authors Biasness"])
        
        if L == R:
            pass
        else:
            ids_to_discard.append(id_)"""

    print("Removing Conflicts")
    conflict = 0 
    conflict_authors_id = []
    for user_l in users_from_left:
        if user_l in users_from_right:
            L = set(data1[data1["Author Id"] == user_l]["Authors Biasness"])
            R = set(data2[data2["Author Id"] == user_l]["Authors Biasness"])
            if L == R:
                pass
            else:
                conflict += 1
                conflict_authors_id.append(user_l)
    print("{} conflicts found: ".format(conflict))

    print("Saving combined Dataset")
    data1.drop(data1[data1["Author Id"].isin(conflict_authors_id)].index, inplace=True)
    data2.drop(data2[data2["Author Id"].isin(conflict_authors_id)].index, inplace=True)

    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)

    # combine the dataframe
    combined_data = pd.concat([data1, data2], ignore_index=True)

    # save the dataframe
    save_csv_file = "data/14. Combined Data revisit.csv"
    combined_data.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("Combined CSV Created!")

    print("Saving training Dataset")

    # training dataset
    left_channel_nn.drop(left_channel_nn[left_channel_nn["Author Id"].isin(conflict_authors_id)].index, inplace=True)
    right_channel_nn.drop(right_channel_nn[right_channel_nn["Author Id"].isin(conflict_authors_id)].index, inplace=True)

    left_channel_nn.reset_index(drop=True, inplace=True)
    right_channel_nn.reset_index(drop=True, inplace=True)

    # combine the dataframe
    training_data = pd.concat([left_channel_nn, right_channel_nn], ignore_index=True)

    save_csv_file = "data/15. Training Dataset revisit.csv"
    training_data.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("Training CSV updated!")

if __name__ == '__main__':
    main()