import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
df = pd.read_csv("Datasets/servo.data", header=None)
df.columns = ['motor', 'screw', 'pgain', 'vgain', 'class']
print(df.describe())

# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
slice_1 = df[df.vgain == 5]
print(len(slice_1))

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
slice_2 = df[(df.motor == 'E') & (df.screw == 'E')]
print(len(slice_2))


# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
slice_3 = df[df.pgain == 4]
print(slice_3.vgain.mean())



# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!
print(df.dtypes)



