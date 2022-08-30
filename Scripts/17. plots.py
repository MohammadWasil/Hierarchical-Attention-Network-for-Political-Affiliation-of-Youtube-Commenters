
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# same for the right
left_channel = pd.read_csv("data/8. comments LEFT.csv")
summation = left_channel["Video Category"].value_counts().sum()

# for left lenaing youtube channel
category = ["News & Politics", "Nonprofits & Activism", "Entertainment", "People & Blogs", "Education", "Comedy"]
plt.figure(figsize=(12,8))

ax = sns.countplot(x="Video Category", order=category, data=left_channel)

for idx, p in enumerate(ax.patches): 
    per = 100 * p.get_height()/summation
    if per < 0.0005:
        percentage = "< 0.0005%"
    else:
        percentage = '{:.2f}%'.format(per)
    x = p.get_x() + p.get_width()/2
    y = p.get_height() + 10000
    ax.annotate(percentage, (x, y+1), horizontalalignment='center', verticalalignment='bottom')
plt.xticks(rotation=90)

plt.title("Distribution of Video Categories for Left Leaning YouTube Channel")
plt.xlabel("Categories")
plt.ylabel("Count")
plt.tight_layout()

plt.savefig('Distribution of Video Category Left.jpg')

plt.show()
