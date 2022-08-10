"""
preprocess the given trainng dataset suitable for training.
"""

import pandas as pd
import os
from utils import DIRECTORY_PATH, dictionary, preprocess

def main():
    # plaese also save vidoes biasness and authors id
    data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/15. Training Dataset revisit.csv"))

    video_biasness = []
    video_Category = []
    author_ids = []
    author_names = []
    author_comments = []
    authors_biasness = []

    number_of_comments = []

    ids = data.groupby('Author Id')["Author Name"].value_counts().index[:]
    for i in ids:
        
        user_id = i[0]
        
        #right_comments.append(R.loc[R['Author Id'] == user_id]["Authors Comment"].tolist())
        author_data = data.loc[data['Author Id'] == user_id]
        
        r_text = ""
        num_com = 0
        comments = author_data["Authors Comment"]
        for comment in comments:
            
            text = preprocess(comment, dictionary)
            r_text = r_text + text + " -|- "
            # count the number of comments in each documents.
            num_com += 1
        number_of_comments.append(num_com)
        
        author_ids.append(user_id)
        author_names.append(list(set(author_data["Author Name"]))[0])
        author_comments.append(r_text)
        authors_biasness.append(list(set(author_data["Authors Biasness"]))[0])
        
        video_biasness.append(list(set(author_data["Biasness"]))[0])
        video_Category.append(list(set(author_data["Video Category"])))
    

    training_data = pd.DataFrame(
            {"Video Biasness" : video_biasness,
            "Video Category": video_Category,
            "Author Id" : author_ids,
            "Author Name" : author_names,
            "Authors Comment" : author_comments,
            "Num of Comments" : number_of_comments,
            "Authors Biasness" : authors_biasness
            })

    save_csv_file = "data/16. Training Dataset revisit.csv"
    training_data.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)

if __name__ == '__main__':
    main()

"""
Statistics
a=training_data[training_data["Video Biasness"] == "RIGHT"]
a["Num of Comments"].sum()

c = a[a["Authors Biasness"] == "LEFT"]
d = a[a["Authors Biasness"] == "RIGHT"]

c["Num of Comments"].sum()
d["Num of Comments"].sum()

b=training_data[training_data["Video Biasness"] == "LEFT"]

c = b[b["Authors Biasness"] == "LEFT"]
d = b[b["Authors Biasness"] == "RIGHT"]

c["Num of Comments"].sum()
d["Num of Comments"].sum()


"""