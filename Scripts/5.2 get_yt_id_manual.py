"""
Manually changing the youtube channel's ids.

Input: 5.1 get_yt_ids.csv
Output: 5.1 get_yt_ids.csv
"""

import pandas as pd
import os
def main():

    DIRECTORY_PATH = "D:/MSc Data Science/Elective Modules - Research Modules/[INF-DS-RMB] Research Module B/RM Code/Sentiment-Classification-Youtube-Comments-Political-Affiliation/"
    channel_yt_twitter = pd.read_csv(os.path.join(DIRECTORY_PATH, "data/5.1 get_yt_ids.csv"))


    channel_yt_twitter.loc[218, "Youtube Channel"] = "UCYWIEbibRcZav6xMLo9qWWw"
    channel_yt_twitter.loc[2271, "Youtube Channel"] = "UC1PRGaduTPEkvzBYAthpkhQ"
    channel_yt_twitter.loc[2205, "Youtube Channel"] = "UCun4tg1BecN4PuxwZ6mL3NA"
    channel_yt_twitter.loc[2231, "Youtube Channel"] = "UCKV_ACaioUGSz0pyg07Zekg"
    channel_yt_twitter.loc[2237, "Youtube Channel"] = "UCkJ1N-7g9Q6n7KnriGit-Ig"
    channel_yt_twitter.loc[2251, "Youtube Channel"] = "UCE1KT9DTZCQtf3PmAmWwOIA"
    channel_yt_twitter.loc[2275, "Youtube Channel"] = "UCWE58e_BbmBgE8h-oskutQA"
    channel_yt_twitter.loc[2568, "Youtube Channel"] = "UCiRXT7Ib7A8VkoJNjfYTctQ"
    channel_yt_twitter.loc[2994, "Youtube Channel"] = "UCgZeFXOjOCNq7QMgTEKNgJQ"
    channel_yt_twitter.loc[3028, "Youtube Channel"] = "UC9gm0OdmoAkGxIZG8veaxnw"

    # UCIbFhmYOqhvrudtVZG5-KEA  -  UCEeLgkduIxQthKGHKSXL32A
    channel_yt_twitter.loc[395, "Youtube Channel"] = "UCIbFhmYOqhvrudtVZG5-KEA"

    #Index:  545
    #UC7ga3FLMFOOpMQwaYoW42bw  -  UCewJXiU0kotMeq4iW9N-Yig
    channel_yt_twitter.loc[545, "Youtube Channel"] = "UC7ga3FLMFOOpMQwaYoW42bw"

    #Index:  724
    #UCuVxaQDraOja6xKidcmoufA  -  UCNB7H1Qqmb0BxmZGcsLTDJw
    channel_yt_twitter.loc[724, "Youtube Channel"] = "UCuVxaQDraOja6xKidcmoufA"

    #Index:  827
    #UC5w-uRIutST34hir01q1iSQ  -  UC8XmG4WEStCycLR_ACPHVgA
    channel_yt_twitter.loc[827, "Youtube Channel"] = "UC5w-uRIutST34hir01q1iSQ"

    #Index:  933
    #UCXJryYh6xcW5iEeJGzK191A  -  UCP36CtKhMCriH9INZQuzt0w
    channel_yt_twitter.loc[933, "Youtube Channel"] = "UCXJryYh6xcW5iEeJGzK191A"

    #Index:  936
    #UCftwRNsjfRo08xYE31tkiyw  -  UCJyB_mF8ym2fc8qO38n_ykA
    channel_yt_twitter.loc[936, "Youtube Channel"] = "UCftwRNsjfRo08xYE31tkiyw"

    #Index:  1024
    #UCb--64Gl51jIEVE-GLDAVTg  -  UCE2LqKjr69KCxPIXP7sZBhQ
    channel_yt_twitter.loc[1024, "Youtube Channel"] = "UCb--64Gl51jIEVE-GLDAVTg"

    #Index:  1035
    #UC83jt4dlz1Gjl58fzQrrKZg  -  UCFr4T8Jf0Kpkb87EIv41eiA
    channel_yt_twitter.loc[1035, "Youtube Channel"] = "UC83jt4dlz1Gjl58fzQrrKZg"

    #Index:  1160
    #UCC69dxCZQB9VURlHQ8wesPA  -  UCpxHilQNhaFSxuPi9j9ZZ9g
    channel_yt_twitter.loc[1160, "Youtube Channel"] = "UCC69dxCZQB9VURlHQ8wesPA"

    #Index:  1223
    #UCbrPqq29C9Q_TQP7OFFRzcw  -  UCM1uzhF6DbRUMKRw9ZeOhpA
    channel_yt_twitter.loc[1223, "Youtube Channel"] = "UCbrPqq29C9Q_TQP7OFFRzcw"

    #Index:  1227
    #UC41xJSNY2xCPC4-gOsRcDcg  -  UCiKLoTpQiqKOf5bwRjg3ACw
    channel_yt_twitter.loc[1227, "Youtube Channel"] = "UC41xJSNY2xCPC4-gOsRcDcg"

    #Index:  1376
    #UCqnkfAaM9RA2uNwTB5MnNrA  -  UCaqxll66pHGJKhC5v8NHO7Q
    channel_yt_twitter.loc[1376, "Youtube Channel"] = "UCqnkfAaM9RA2uNwTB5MnNrA"

    #Index:  1488
    #UCuwdoOuo2MLU2ebspAWKcxQ  -  UCbSl9jBgZ2MqLl6Q_3IZ6IQ
    channel_yt_twitter.loc[1488, "Youtube Channel"] = "UCuwdoOuo2MLU2ebspAWKcxQ"

    #Index:  1503
    #UCVSNOxehfALut52NbkfRBaA  -  UCbeTcyLef06QJSHuj8jlzvg
    channel_yt_twitter.loc[1503, "Youtube Channel"] = "UCVSNOxehfALut52NbkfRBaA"

    #Index:  1516
    #UCb--64Gl51jIEVE-GLDAVTg  -  UCE2LqKjr69KCxPIXP7sZBhQ
    channel_yt_twitter.loc[1516, "Youtube Channel"] = "UCb--64Gl51jIEVE-GLDAVTg"

    #Index:  1749
    #UCNFGSWVOdVWEe9XJNnfTdyQ  -  UCseeI4EanP9fMdemTV6G4SA
    channel_yt_twitter.loc[1749, "Youtube Channel"] = "UCNFGSWVOdVWEe9XJNnfTdyQ"

    #Index:  1779
    #UC_vt34wimdCzdkrzVejwX9g  -  UCWA_amClqKHpE6ZWiPL7bdA
    channel_yt_twitter.loc[1779, "Youtube Channel"] = "UC_vt34wimdCzdkrzVejwX9g"

    #Index:  1857
    #UCb1Ti1WKPauPpXkYKVHNpsw  -  UCh6mcAlBy6v0GOdqRmjwuWQ
    channel_yt_twitter.loc[1857, "Youtube Channel"] = "UCb1Ti1WKPauPpXkYKVHNpsw"

    save_csv_file = "data/5.1 get_yt_ids.csv"
    channel_yt_twitter.to_csv(os.path.join(DIRECTORY_PATH, save_csv_file), encoding='utf-8', index=False)
    print("CSV updated!")

if __name__ == '__main__':
    main()