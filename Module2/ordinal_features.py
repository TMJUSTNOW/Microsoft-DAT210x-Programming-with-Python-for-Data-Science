import pandas as pd

ordered_satisfaction = ['Very Unhappy', 'Unhappy', 'Neutral', 'Happy', 'Very Happy']
df = pd.DataFrame({'satisfaction': ['Mad', 'Happy', 'Unhappy', 'Neutral']})

df.satisfaction = df.satisfaction.astype("category",
                                         ordered=True,
                                         categories=ordered_satisfaction
                                         ).cat.codes

print(df)
df = pd.DataFrame({'vertebrates': [
    'Bird',
    'Bird',
    'Mammal',
    'Fish',
    'Amphibian',
    'Reptile',
    'Mammal',
]})

# Method 1)
df['vertebrates'] = df.vertebrates.astype("category").cat.codes

print(df)

# Method 2)
df = pd.get_dummies(df,columns=['vertebrates'])

print(df)