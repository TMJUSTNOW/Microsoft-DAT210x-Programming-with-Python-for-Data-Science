import pandas as pd

df = pd.read_csv("Datasets/direct_marketing.csv")

print(df.describe())
#print(df['recency'])
print(df[['recency']])
df.loc[:, 'recency']

# Produces a series object:
df.recency
df['recency']
df.loc[:, 'recency']
df.iloc[:, 0]
df.ix[:, 0]

#
# Produces a dataframe object:
df[['recency']]
df.loc[:, ['recency']]
df.iloc[:, [0]]

print(df.iloc[0:5,:3])

print(df[ (df.recency > 10) & (df.newbie == 0)])

print(df.loc[0:4, ['recency', 'channel']])

