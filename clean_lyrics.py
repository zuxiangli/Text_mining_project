import spacy
from spacy.matcher import Matcher
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm",disable=["tagger", "parser"])

lyrics_df=pd.read_csv("all_lyrics.csv").astype(str)

lyrics_df=lyrics_df[lyrics_df["genre"]!="nan"]
lyrics_df=lyrics_df[lyrics_df["lyrics"]!="nan"]

def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_stop == False and token.is_alpha == True])

print(lyrics_df.iloc[1].lyrics)
print(preprocess(lyrics_df.iloc[1].lyrics))

clean_lyrics=pd.DataFrame()

for x in lyrics_df.itertuples():
    #print(x)
    all_words = re.sub(r'[\(\[].*?[\)\]]', '', x[4])
    data={"name": x[1], "artist": x[2], "lyrics": preprocess(x[3]),"genre":x[4]}
    new_df=pd.DataFrame(data,index=[0])
    clean_lyrics=pd.concat((clean_lyrics,new_df))


max([len(x) for x in clean_lyrics.lyrics ])

clean_lyrics=pd.read_csv("cleaned_lyrics.csv")

test_lyrics=pd.DataFrame()
for x in clean_lyrics.itertuples():
    data={"name": x[1], "artist": x[2], "lyrics": " ".join(set(x.lyrics.split())),"genre":x[4]}
    new_df=pd.DataFrame(data,index=[0])
    test_lyrics=pd.concat((test_lyrics,new_df))

