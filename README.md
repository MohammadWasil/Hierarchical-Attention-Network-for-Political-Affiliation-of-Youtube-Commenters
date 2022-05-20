# Sentiment-Classification-Youtube-Comments-Political-Affiliation

To scrap the data:
1. Change the API key and tokens in "config_KEYS EXAMPLE.yml" with your own twitter API key and tokens
2. Run 
```{python}
python 'Scrap MBFC website.py'
```
This will create a list of News channels with their website link and country.
3. Then run
```{python}
python '2. scrap_youtube_twitter.py'
```
This will scrap each News website with their youtube channel name and twitter handle.
