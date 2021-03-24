from bs4 import BeautifulSoup
import requests
import requests.exceptions
from urllib.parse import urlsplit
from collections import deque
import re
from lxml import etree
import pandas as pd

years=range(1970,2011)
genres=["top-100-songs","rock","country","rnb"]

resDF=pd.DataFrame()
for genre in genres:
    for year in years:
        baseUrl='https://playback.fm/charts/'
        url=baseUrl+"/"+genre+"/"+str(year)
        #url = 'https://playback.fm/charts/country/1987'
        strhtml = requests.get(url).content

        soup=BeautifulSoup(strhtml,'lxml')

        songnames=[]
        artistnames=[]

        data=soup.select("span[class='song']")
        for item in data:
            songnames.append(item.get_text().strip())


        data=soup.select("a[class='artist']")
        for item in data:
            artistnames.append(item.get_text().strip())
        if genre=="top-100-songs":
            genre="pop"
        track=zip(songnames,artistnames,[year]*100,[genre]*100)
        tmp_df=pd.DataFrame(track)
        resDF=pd.concat([resDF,tmp_df])

resDF.to_csv("data.csv",index=False)
