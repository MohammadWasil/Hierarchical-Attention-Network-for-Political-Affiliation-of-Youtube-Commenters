"""
Scrap MBFC website for the lists of news channels, with their names, website, their biasness.
Output: News channels.csv
"""

import time, requests, re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

import pandas as pd

headers = requests.utils.default_headers()

headers.update(
    {
        'User-Agent': 'My User Agent 1.0',
    }
)

ideology_pages = ['left', 'leftcenter', 'center', 'right-center', 'right', 'fake-news']
bias_rating = ['left', 'center-left', 'center', 'center-right', 'right', 'fake-news']

links = []
names = []
bias_ratings = []

# statistics
Number_of_news_channels = 0

for i, pages in enumerate(ideology_pages):
    tags_ignored = 0
    
    MBFC_SOURCE_URL = "https://mediabiasfactcheck.com/{}/".format(pages)
    #print(MBFC_SOURCE_URL)
    response = requests.get(MBFC_SOURCE_URL)
    soup = BeautifulSoup(response.text, 'lxml')

    #list_pages = soup.find( "table", {"id":"mbfc-table"} ).find_all("tr")
    table = soup.find('table', attrs={'id':'mbfc-table'}).find("tbody")
    list_rows = table.find_all('tr')

    for row_idx, row in enumerate(list_rows):
        try:
            link = row.find('td').find('a', href=True).get('href')
            name = row.find('td').text
            
            names.append(name)
            links.append(link)
            bias_ratings.append(bias_rating[i])
            Number_of_news_channels += 1
            
        except:
            tags_ignored += 1
            #print("ignoring one tag")
    #news_channel_list = news_channel_list.reset_index(drop=True)
    print("Tags ignored for {} ideology news channels list: {}".format(pages, tags_ignored))

news_channel_list = pd.DataFrame(
    {'Name': names,
     'Link': links,
     'Bias Rating': bias_ratings
    })

news_channel_list.to_csv("News channels version 1.csv", encoding='utf-8', index=False)
print("Number of news channels collected: {}".format(Number_of_news_channels))

# statistics for channel country and their website links.
# Number of news channels for which the stats country is not available - country_NA
# Number of news channels for which the stats source is not available - source_Na
country_NA = 0
source_NA = 0

print("Requesting individual websites on MBFC list")
# requesting each websites
for j, link in enumerate(news_channel_list["Link"]):
    #link = "https://mediabiasfactcheck.com/american-united-separation-church-state/"
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'lxml')

    # get the p tag with Bias Rating and country.
    news_info = soup.find("div", {"class" : "entry-content clearfix"})

    try:
        news_channel_information = news_info.find_all(text = re.compile('Country:'))[0].parent.get_text(strip=False).split("\n")

        for n_i in news_channel_information:
            if n_i.split(":")[0] == "Country":
                country = n_i.split(":")[1].strip()
                if country != "":
                    #country.append(n_i.split(":")[1].strip() )
                    news_channel_list.loc[j, "Country"] = country
                else:
                    #print("country is empty")
                    raise 

        print("News channel Index : ", j, " Done!")
    except:
        country_ = "NA"
        country_pattern = r'Country:\s[a-zA-Z]+'
        try:
            for line in news_info.text.split('\n'):
                if bool(re.match(country_pattern, line)):
                    #country = line.rstrip().split(':')[1].split('(')[0].strip()
                    country_ = line.split(":")[1].strip()
                    #country.append(country_)
                    news_channel_list.loc[j, "Country"] = country_

            if country_ == "NA":
                country_NA += 1
                print("News channel Index : ", j, " Not Done!")
            else:
                print("News channel Index : ", j, " Done by 2nd method!")
        except:
            country_NA += 1
            print("The MBFC website for {} does not exists!".format(news_channel_list["Name"][j]))

    try:
        source = news_info.find_all(text = re.compile('Source:'))[0].parent.find('a', href=True).get('href')
        #news_website.append(source)
        news_channel_list.loc[j, "Website"] = source
        print("News Channel source for Index : ", j, " Done!")

    except:
        source_ = "NA"
        source_pattern = r'Sources?:\shttps?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        try:
            for line in news_info.text.split('\n'):
                if bool(re.match(source_pattern, line)):
                    source_ = line.split(":", 1)[1].strip().rstrip(" ")
                    #news_website.append(source_)
                    news_channel_list.loc[j, "Website"] = source_

            if source_ == "NA":
                source_NA += 1
                print("News channel source Index : ", j, " Not Done!")
            else:
                print("News channel source Index : ", j, " Done by 2nd method!")
        except:
            source_NA += 1
            print("The MBFC website for {} does not exists!".format(news_channel_list["Name"][j]))

print("Number of news channels without their country : {}".format(country_NA))
print("Number of news channels without their sources/link : {}".format(source_NA))

news_channel_list.to_csv("1. News channels.csv", encoding='utf-8', index=False)