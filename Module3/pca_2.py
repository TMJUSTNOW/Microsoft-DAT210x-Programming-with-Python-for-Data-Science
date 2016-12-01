import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')


pca = PCA(n_components=2)
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)

T = pca.transform(df)

df.shape
(430, 6)  # 430 Student survey responses, 6 questions..

T.shape
(430, 2)  # 430 Student survey responses, 2 principal components..
