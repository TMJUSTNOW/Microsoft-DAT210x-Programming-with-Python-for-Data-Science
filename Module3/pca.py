import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')
# If the above line throws an error, use plt.style.use('ggplot') instead

student_dataset = pd.read_csv("Datasets/student-mat.data", delimiter=';')
print(student_dataset.columns)
my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']]


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('G1')
ax.set_ylabel('G2')
ax.set_zlabel('G3')
ax.scatter(student_dataset.G1, student_dataset.G2, student_dataset.G2, c='r', marker='.')
plt.show()
plt.imshow(my_dataframe.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(my_dataframe.columns))]
plt.xticks(tick_marks, my_dataframe.columns, rotation='vertical')
plt.yticks(tick_marks, my_dataframe.columns)
plt.show()
my_dataframe.plot.scatter(x='G1', y ='G2')
my_dataframe.plot.scatter(x='G1', y ='G3')
my_dataframe.plot.scatter(x='G2', y ='G3')
plt.show()

#PCA starts here

pca = PCA(n_components=2, whiten=False)
pca.fit(my_dataframe)

T = pca.transform(my_dataframe)
print(T.shape)
print(T)
