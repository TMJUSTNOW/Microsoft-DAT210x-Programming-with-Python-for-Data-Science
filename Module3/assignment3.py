import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
data_frame = pd.read_csv('Datasets/wheat.data')


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
ax = Axes3D(fig)
ax.set_xlabel('Area')
ax.set_ylabel('Perimeter')
ax.set_zlabel('Asymmetry')

ax.scatter(data_frame.area, data_frame.perimeter, data_frame.asymmetry, c='r', marker='.')


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes

ax = Axes3D(fig)
ax.set_xlabel('Width')
ax.set_ylabel('Groove')
ax.set_zlabel('Length')

ax.scatter(data_frame.width, data_frame.groove, data_frame.length, c='green', marker='.')


plt.show()


