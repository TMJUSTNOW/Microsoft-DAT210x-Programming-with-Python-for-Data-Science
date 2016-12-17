import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import random, math
import glob, os
# Uses the Image module (PIL)
from scipy import misc
from sklearn import manifold

# Look pretty...
matplotlib.style.use('ggplot')




#
# Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
samples = []

#
# Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
colors= []
path = "./Datasets/ALOI/32/"
for file in os.listdir(path):
    if file.endswith(".png"):
        img = misc.imread(path+file)
        img = img.reshape(-1)
        samples.append(img)
        colors.append('b')
print(len(samples))

#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
path = "./Datasets/ALOI/32i/"
for file in os.listdir(path):
    if file.endswith(".png"):
        img = misc.imread(path+file)
        img = img.reshape(-1)
        samples.append(img)
        colors.append('r')

df = pd.DataFrame.from_records(samples)
print(df.describe())

iso = manifold.Isomap(n_neighbors=6, n_components=3)
print("iso map fit start ")
iso.fit(df)
print("iso map fit end ")
manifold = iso.transform(df)
print(df.shape)
print(manifold.shape)
print(manifold[0])

# Render the 2D isomap component
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('ISOMAP 3S')
ax.scatter(manifold[:,0], manifold[:,1], c=colors, marker='.', alpha=0.75)

# Render the 3D isomap component
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('ISOMAP 3S')
ax.scatter(manifold[:,0], manifold[:,1], manifold[:,2], c=colors, marker='.', alpha=0.75)

plt.show()
exit()


#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

