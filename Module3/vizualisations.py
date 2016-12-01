import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot') # Look Pretty
# If the above line throws an error, use plt.style.use('ggplot') instead

student_dataset = pd.read_csv("Datasets/student-por.data", delimiter=';')

my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5)
my_dataframe.plot.hist(alpha=0.5)
student_dataset.plot.scatter(x='G1', y ='G3')

fig = plt.figure()
ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grate')
ax.set_ylabel('First Grate')
ax.set_zlabel('Daily Alcohol')

ax.scatter(student_dataset.G1, student_dataset.G3, student_dataset['Dalc'], c='r', marker='.')
plt.show()
