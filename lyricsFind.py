import lyricsgenius
genius = lyricsgenius.Genius("Ggzwsp6RJf7Jy9-nh1xM8hw92OXJoWJygdAZfW76yArZZzfCYJL0OfVXPJcTVCfR")
genius.response_format = 'plain'
import pandas as pd

df=pd.read_csv("hiphop_data.csv")
lyrics_df=pd.DataFrame()


for x in df.itertuples():
    print(x[0])
    try:
        song = genius.search_song(x[1], x[2])
        lyrics = song.lyrics
        data = {"index": x[0], "name": x[1], "artist": x[2], "lyrics": lyrics}

    except:
        data = {"index": x[0], "name": x[1], "artist": x[2], "lyrics": "Not exist"}
        continue
    tmp_df = pd.DataFrame(data, index=[0])
    lyrics_df=pd.concat([lyrics_df,tmp_df])


#filename = "lyrics+" + str(i) + ".csv"
lyrics_df.to_csv("hip_hop_lyrics.csv", index=False)
lyrics_df = pd.DataFrame()


lyrics_df=pd.read_csv("hip_hop_lyrics.csv")
