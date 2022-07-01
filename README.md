# Sentiment-Classification-Youtube-Comments-Political-Affiliation

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMohammadWasil%2FSentiment-Classification-Youtube-Comments-Political-Affiliation&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

### Packages
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
