"""
Create first layer of annotated data for han training (this was done using users subscription data) - This was just the sample
Annotating hashtags took time, so inbetwen this, we crrate annotated dataset (from subscription list) to create our HAN model.
"""

import pandas as pd
import os
from utils import DIRECTORY_PATH, dictionary, preprocess

def main():

    left = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/10. comments LEFT.csv"))
    right = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/10. comments RIGHT.csv"))

    # get all non null rows of column Authors Biasness
    left = left[left["Authors Biasness"].notnull()]
    right = right[right["Authors Biasness"].notnull()]

    merged = left.append(right, ignore_index=True)

    # get all non null rows of column Authors Biasness
    merged = merged[merged["Authors Biasness"].notnull()]

    # get those column which have exactly 5 authors commenting
    #v = left["Author Id"].value_counts()
    #l = left[left["Author Id"].isin(v.index[v.eq(5)])]

    # distribute into left and right authors. Mind here, it is not channels, but Authors.
    R = merged[merged["Authors Biasness"] == "RIGHT"]
    L = merged[merged["Authors Biasness"] == "LEFT"]

    # comments from conservative
    right_comments = []
    right_num_of_comments = []
    ids = R.groupby('Author Id')["Author Name"].value_counts().index[:]
    for i in ids:
        user_id = i[0]
        #right_comments.append(R.loc[R['Author Id'] == user_id]["Authors Comment"].tolist())
        comments = R.loc[R['Author Id'] == user_id]["Authors Comment"]
        r_text = ""
        right_num_of_comment = 0
        for comment in comments:
            text = preprocess(comment, dictionary)
            r_text = r_text + text + " -|- "
            # count the number of comments in each documents.
            right_num_of_comment += 1
        
        right_comments.append(r_text)
        right_num_of_comments.append(right_num_of_comment)
        
    r_annot = ["RIGHT"] * len(right_comments)

    # comments from liberals
    left_comments = []
    left_num_of_comments = []
    ids = L.groupby('Author Id')["Author Name"].value_counts().index[:]
    for i in ids:
        user_id = i[0]
        #right_comments.append(R.loc[R['Author Id'] == user_id]["Authors Comment"].tolist())
        comments = L.loc[L['Author Id'] == user_id]["Authors Comment"]
        l_text = ""
        left_num_of_comment = 0
        for comment in comments:
            text = preprocess(comment, dictionary)
            l_text = l_text + text + " -|- "
            left_num_of_comment += 1
        
        left_comments.append(l_text)
        left_num_of_comments.append(left_num_of_comment)
    l_annot = ["LEFT"] * len(left_comments)

    comments = pd.DataFrame(
        {"comment" : right_comments + left_comments,
        'Annot': r_annot + l_annot,
        "Number of Comment" : right_num_of_comments + left_num_of_comments
        })

    save_csv_file = "data/12. Subscription Training Data.csv"
    comments.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)

if __name__ == '__main__':
    main()


