import pandas as pd
import html5lib


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
url = "http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015"
df = pd.read_html(url)[0]


# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
df.columns = [ "RK",'PLAYER', 'TEAM', 'GP', 'G', 'A','PTS','PLUS_MINUS' , 'PIM', 'PTS_PER_GAME', 'SOG', 'PCT', 'GWG',	'G_PP', 'A_PP', 'G_SH', 'A_SH']


# TODO: Get rid of any row that has at least 4 NANs in it
df = df.dropna(axis=0, thresh=4)


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
df = df.drop(df[df.PLAYER =='PLAYER'].index, axis = 0)


# TODO: Get rid of the 'RK' column
#
df = df.drop( labels=['RK'], axis = 1)


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df = df.reset_index(drop=True)
print(df)


# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
df.GP = pd.to_numeric(df.GP, errors='coerce')
df.G = pd.to_numeric(df.G, errors='coerce')
df.A = pd.to_numeric(df.A, errors='coerce')
df.PTS = pd.to_numeric(df.PTS, errors='coerce')
df.PLUS_MINUS = pd.to_numeric(df.PLUS_MINUS, errors='coerce')
df.PIM = pd.to_numeric(df.PIM, errors='coerce')
df.PTS_PER_GAME = pd.to_numeric(df.PTS_PER_GAME, errors='coerce')
df.SOG = pd.to_numeric(df.SOG, errors='coerce')
df.PCT = pd.to_numeric(df.PCT, errors='coerce')
df.GWG = pd.to_numeric(df.GWG, errors='coerce')
df.G_PP = pd.to_numeric(df.G_PP, errors='coerce')
df.A_PP = pd.to_numeric(df.A_PP, errors='coerce')
df.G_SH = pd.to_numeric(df.G_SH, errors='coerce')
df.A_SH = pd.to_numeric(df.A_SH, errors='coerce')

print(df.dtypes)
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#After completing the 6 steps above, how many rows remain in this dataset? (Not to be confused with the index!)
print(len(df))
#How many unique PCT values exist in the table?
print(len(df.PCT.unique()))
#What is the value you get by adding the GP values at indices 15 and 16 of this table?
gp_sum = df.GP[15] + df.GP[16]
print(gp_sum)



