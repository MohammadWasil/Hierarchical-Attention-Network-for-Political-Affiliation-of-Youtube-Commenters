"""
preprocess the un-annotated dataset for inference.
"""
import pandas as pd
import os
from utils import preprocess, dictionary, DIRECTORY_PATH

def main():
    LEANING = "LEFT"
    #data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/For Inference Youtube {} Channel.csv".format(LEANING)))

    data = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/13. comments {} with Hashtag annotations.csv".format(LEANING)), 
                        lineterminator='\n')
    unlabel_data = data[data['Authors Biasness'].isnull()]

    ids = unlabel_data.groupby('Author Id')["Author Name"].value_counts().index[:]
    
    save_ids = []
    output_file = "data/Track the ids - {} revisit.txt".format(LEANING)
    if os.path.exists(os.path.join(DIRECTORY_PATH, output_file)):
        with open(os.path.join(DIRECTORY_PATH, output_file), 'r') as fin:
            for line in fin:
                save_ids.append(line[:-1])
            print(save_ids)
            print("Processed {} authors!!!".format(len(save_ids)))
    
    counter = 0
    print("Total ids to process: {}".format(len(ids)))
    for i in ids:
        user_id = i[0]
        if user_id not in save_ids:
            counter += 1

            author_data = unlabel_data.loc[unlabel_data['Author Id'] == user_id]

            #right_comments.append(R.loc[R['Author Id'] == user_id]["Authors Comment"].tolist())
            r_text = ""
            num_com = 0
            comments = author_data["Authors Comment"]
            for comment in comments:
                text = preprocess(comment, dictionary)
                r_text = r_text + text + " -|- "
                # count the number of comments in each documents.
                num_com += 1
                        
            inference_data = pd.DataFrame(
                    {"Video Biasness" : list(set(author_data["Biasness"]))[0],
                    "Video Category": list(set(author_data["Video Category"])),
                    "Author Id" : user_id,
                    "Author Name" : list(set(author_data["Author Name"]))[0],
                    "Authors Comment" : r_text,
                    "Num of Comments" : num_com,
                    "Authors Biasness" : list(set(author_data["Authors Biasness"]))[0]
                    })

            # Write the new data to the CSV file in append mode
            inference_data.to_csv(os.path.join(DIRECTORY_PATH, 'data/For Inference Youtube {} Channel revisit.csv'.format(LEANING)), 
                                    mode='a', header=False, index=False)
            if counter % 100 == 0:
                print("Counter: ", counter)

            # save the id
            save_ids.append(i)
            with open(os.path.join(DIRECTORY_PATH, output_file), 'a') as fin:
                fin.write(f"{user_id}\n")

if __name__ == '__main__':
    main()