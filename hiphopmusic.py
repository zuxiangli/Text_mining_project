from bs4 import BeautifulSoup
import requests
import requests.exceptions
from urllib.parse import urlsplit
from collections import deque
import re
from lxml import etree
import pandas as pd
import time

years=range(2002,2020,2)
resDF=pd.DataFrame()
#resDF=pd.read_csv("hiphop_data.csv")
for year in years:
    for year in range(year,year+2):
        print(year)
        baseUrl = "https://www.billboard.com/charts/year-end/"+str(year)+"/hot-r-and-and-b-hip-hop-songs"
        strhtml = requests.get(baseUrl).content

        soup = BeautifulSoup(strhtml, 'lxml')

        songnames = []
        artistnames = []

        data=soup.select("div[class='ye-chart-item__title']")
        for item in data:
            songnames.append(item.get_text().strip())

        data=soup.select("div[class='ye-chart-item__artist']")
        for item in data:
            artistnames.append(item.get_text().strip())

        track = zip(songnames, artistnames, [year] * 100, ["hiphop"] * 100)
        tmp_df = pd.DataFrame(track)
        resDF = pd.concat([resDF, tmp_df])
        time.sleep(5)

resDF.to_csv("hiphop_data.csv",index=False)

resDF=pd.read_csv("hiphop_data.csv")