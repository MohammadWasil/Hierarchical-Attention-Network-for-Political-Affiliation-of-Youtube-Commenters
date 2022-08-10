# Sentiment-Classification-Youtube-Comments-Political-Affiliation

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadWasil%2FSentiment-Classification-Youtube-Comments-Political-Affiliation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

### Installing Dependencies
Install package `scrapetube` using pip to scrap the youtube videos id:
```
pip install scrapetube
pip install youtubesearchpython
```

Change the name of the file `config_KEYS EXAMPLE.yml` to `config_KEYS.yml` and add your twitter credentials there.

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

16. Create first layer of annotated data for han training (this was done using users subscription data)
```{python}
python '12. create_data_subscription_training.py'
```



### (Annotated) Dataset Description

#### Original Paper
| Leaning | Users (from Subscription list) | Users (from Hasgtags) | Available |
| --- | --- | --- | --- |
| Liberal | 61320 | 8616 | Yes* |
| Conservatives | 86134 | 8144 | Yes* |

* But nor scrapable. USers_id has been hashed to some other numbers. Comments scraping is not reproducable.


#### Our Dataset (Till now, exlucding Annotations from hashtags)
| Leaning | Users (from Subscription list) | Users (from Hasgtags) | Available |
| --- | --- | --- | --- |
| Liberal | 7,714 |  | --- |
| Conservatives | 4,975 |  | --- |

### Dataset from External sources
| Paper | Dataset | Available | Size (#Comments) | Github |
| --- | --- | --- | --- | --- |
| Lewis 2018 | Big data with comments | Yes | Annotations not Available | https://github.com/RSButner/Alt_Inf_Net |
| Rebeiro 2020 | Not so big | Yes | Annotations not Available (No comments) Data might be avaiblae upon request | https://github.com/manoelhortaribeiro/radicalization_youtube |
| Ledwich and Zaitsev 2020 | - | No | Data not available | - |

### Model Result

| Model | Validation F1 Score  | Validation Loss | Validation Accuracy | Test F1 Score | Test Loss | Test Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| HAN w/ Embedding | --- | --- | --- | --- | --- | --- |
| HAN w/o embedding | - | - | - | - | - | - |
| LSTM w/ Embedding | 0.83 | 0.439 | 86.83 | 0.83 | 0.45 | 86.05 |
| LSTM w/o embedding | 0.80 | 0.468 | 84.11 | 0.79 | 0.48 | 82.56 |




