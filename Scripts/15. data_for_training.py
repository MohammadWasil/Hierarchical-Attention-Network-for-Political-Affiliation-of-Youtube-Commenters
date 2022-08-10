"""
preprocess the given trainng dataset suitable for training.
"""

import pandas as pd
import os
from utils import DIRECTORY_PATH, dictionary, preprocess

def main():
    # plaese also save vidoes biasness and authors id
    data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/15. Training Dataset.csv"))

    # distribute into left and right
    R = data[data["Authors Biasness"] == "RIGHT"]
    L = data[data["Authors Biasness"] == "LEFT"]

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

    save_csv_file = "data/16. Training Dataset.csv"
    comments.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)

if __name__ == '__main__':
    main()