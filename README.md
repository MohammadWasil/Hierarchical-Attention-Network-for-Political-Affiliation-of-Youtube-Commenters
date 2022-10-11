# Hierarchical Attention Network for Political Affiliation of Youtube Commenters

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadWasil%2FSentiment-Classification-Youtube-Comments-Political-Affiliation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

The research project was associated with "[INF-DS-RMB] Research Module B: Projekt: Social Media and Business Analytics Project", Summer Semester 2022, for my Masters of Science: Data Science, University of Potsdam, Germany. Associated Research Paper can be found [here](https://www.researchgate.net/publication/364308344_Political_Affiliation_of_YouTube_Commenters_with_Hierarchical_Attention_Network).

If you want to train/inference/visualize Hierarchical Attention Model (HAN) or LSTM, follow the steps shown in readme [here](https://github.com/MohammadWasil/Sentiment-Classification-Youtube-Comments-Political-Affiliation/tree/main/Model%20Cod#training-description)

### Installing Dependencies for Data Scraping
Install package `scrapetube` using pip to scrap the youtube videos id:
```
pip install scrapetube
pip install youtubesearchpython
```
Also, create twitter api key and Youtube-api key for scraping data.

Add your twitter credentials in `utility/config_KEYS.yml`

To scrap the data:
1. Change the API key and tokens in `config_KEYS.yml` with your own twitter API key and tokens
2. To create a list of News channels with their website link and country, Run
```{python}
python 'Scrap MBFC website.py'
```
3. To scrap each News channel website with their Youtube Channel name and twitter handle, run
```{python}
python '2. scrap_youtube_twitter.py'
```
4. Review Twitter handles manually.
```{python}
python '2_1. review_twitter_handle.py'
```
5. To find twitter handle of the remaining news channels, we will do exhaustive search using:
```{python}
python '3. Get_twitter_handle.py'
```
6. To search the youtube search page with channel title to find their youtube channel
```{python}
python '4. Scrap_youtube_channel.py'
```
7. To validate the given youtube channel Id. If youtube channel username is available to us, we will get their youtube channel ID.
```{python}
python '5. scrap_youtube_id.py'
```
8. Another method to get the channel id using their username
```{python}
python '5.1 get_yt_id.py'
```
9. Manually changing the youtube channel's ids.
```{python}
python '5.2 get_yt_id_manual.py'
```
10. From the youtube channel's playlist, get all those videos which was published from 2021-01-01 to 2021-08-31. 
```{python}
python '6. get_yt_channel_playlists_videos.py'
```
11. Using the video Id's scraped in step 10, use those video id's to scrap their comments.
```{python}
python '7. get_yt_comments.py'
```

--- A utility file to combine right and right center files.
```{python}
python '7.1 combine_right_and_center_right_data.py'
```

12. Convert the data from json format to csv.
```{python}
python '8. json_to_csv.py'
```

13. Get the subscription list of all the auhtors who have made comments using Youtube API.
```{python}
python '9. get_authors_subscription.py'
```

14. First step of Annotation. Annotating users as liberals or conservatives using users subscription data and homogeneity score.
```{python}
python '10. user_subscription_homogeneity_score.py'
```

15. Create sepearte dataframe to easily annotate hashtags as being used by liberals or conservatives.
```{python}
python '11. create_df_hashtag_annotations.py'
```

16. Create first layer of annotated data for training (this was done using users subscription data) - Just a sample file to create out models.
```{python}
python '12. create_data_subscription_training.py'
```

17. Second step of Annotation. Annotating users as liberals or conservatives using hashatags used by the user on their comment and homogeneity score.
```{python}
python '13. create_data_from_hashtags.py'
```

18. Find conflicted users (user both in left and right channels with different leaning) and remove them. Then combine both the dataet and save. Next, take those samples, where we know the leaning of the user, and generate annotated training data for training.
```{python}
python '14. generate_training_data revisit.py'
```

19. Preprocess the given trainng dataset (created in step 14 and 17) suitable for training.
```{python}
python '15. data_for_training_revisit.py'
```

20. Preprocess the un-annotated dataset for inference.
```{python}
python '16. null_comments.py'
```

21. Create Plots.
```{python}
python '17. plots.py'
```

22. This can only be executed after yuo have infernece files. On removing conflicts from inferenced result.
```{python}
python '18. remove_conflicts_inference.py'
```

`utils.py` contains all utility functions and variables.
