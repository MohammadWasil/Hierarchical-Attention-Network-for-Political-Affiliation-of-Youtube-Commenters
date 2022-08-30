import pandas as pd

left_channel = pd.read_csv("data/For Inference Youtube LEFT Channel revisit.csv")
right_channel = pd.read_csv("data/For Inference Youtube RIGHT Channel revisit.csv")

# merge them
inference_channels = [left_channel, right_channel]
inference_channels = pd.concat(inference_channels) # merge left and right channel together

# get the duplicates
duplicated = inference_channels[inference_channels.duplicated('Author Id', keep=False) == True] 

# remove those duplicates from main df - inference_channels, and then save
inference_channels = inference_channels[~inference_channels["Author Id"].isin(duplicated["Author Id"])] 
inference_channels = inference_channels.reset_index(drop=True)
inference_channels.to_csv("data/inference part 1.csv", index=False)


# save the duplcated  - to be used after the inference on "duplicated_csv"
duplicated = duplicated.reset_index(drop=True)
duplicated.to_csv("data/duplicated for future.csv", index=False)

# merge the duplicated rows together in "duplicate" dataframe
duplicated = pd.read_csv("data/duplicated for future.csv")
duplicate = duplicated

# drop them
duplicate.drop(['Video Biasness', 'Video Category'], axis=1, inplace=True)  

# merge together the comments, and add the num of comments
duplicated_csv = duplicate.groupby('Author Id').agg({"Author Name":'first',
                                    "Authors Comment":lambda x: "".join(x), 
                                    "Num of Comments": lambda x: sum(x),
                                    "Authors Biasness":'first'}).reset_index()

duplicated.to_csv("data/inference part 2.csv", index=False)

"""
1. After inference part 1, get the result and update the "inference part 1.csv" file.
2. Then calculate data statistics.
3. After inference part 2, get the result, and update the "duplicated for future.csv" file.
4. Then calculate data statistics.
5. Merge both data statistics.
"""

# After inference part 1
import pickle
with open('data/inference part 1 authors biasness.pkl', 'rb') as f:
    inference_1_auth_biasness = pickle.load(f)

inference_channels['Authors Biasness'] = inference_channels['Authors Biasness'].fillna(inference_channels['Author Id'].astype(str).map(inference_1_auth_biasness))
inference_channels.to_csv("data/inference part 1 with annotation.csv", index=False)

# After inference part 2
import pickle
with open('data/inference part 2 authors biasness.pkl', 'rb') as f:
    inference_2_auth_biasness = pickle.load(f)

duplicated['Authors Biasness'] = duplicated['Authors Biasness'].fillna(duplicated['Author Id'].astype(str).map(inference_2_auth_biasness))
duplicated.to_csv("data/inference part 2 with annotation.csv", index=False)
