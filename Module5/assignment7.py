import pandas as pd
from sklearn import preprocessing, manifold
from sklearn.cross_validation  import train_test_split
from sklearn.decomposition import PCA
import sklearn.manifold
from sklearn.neighbors import KNeighborsClassifier
# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:



def plotDecisionBoundary(model, X, y):
  print("Plotting...")
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()



def initialize_scalers_map(X):
  #
  #  Experiment with the basic SKLearn preprocessing scalers. We know that
  # the features consist of different units mixed in together, so it might be
  # reasonable to assume feature scaling is necessary. Print out a description
  # of the dataset, post transformation.

  scalers = dict()
  scalers['StandardScaler'] = preprocessing.StandardScaler().fit(X)
  scalers['MinMaxScaler'] = preprocessing.MinMaxScaler().fit(X)
  scalers['MaxAbsScaler'] = preprocessing.MaxAbsScaler().fit(X)
  scalers['Normalizer'] = preprocessing.Normalizer().fit(X)
  scalers['RobustScaler'] = preprocessing.RobustScaler().fit(X)
  scalers['NoScaler'] = None

  return scalers


# returns X_train and X_test in 2D using PCA or ISOMAP
def dimensionality_reduction(X_train, X_test, Test_PCA = True):
  #
  # PCA and Isomap are your new best friends
  model = None
  if Test_PCA:
    print("Computing 2D Principle Components")
    #
    # Implement PCA here. save your model into the variable 'model'.
    # You should reduce down to two dimensions.
    #
    model = PCA(n_components=2)


  else:
    print("Computing 2D Isomap Manifold")
    #
    # Implement Isomap here. save your model into the variable 'model'
    # Experiment with K values from 5-10.
    # You should reduce down to two dimensions.
    #
    model = manifold.Isomap(n_components=2, n_neighbors=8)

  #
  # Train your model against data_train, then transform both
  # data_train and data_test using your model. You can save the results right
  # back into the variables themselves.
  #
  model.fit(X_train)
  X_train = model.transform(X_train)
  X_test = model.transform(X_test)
  return X_train, X_test



def process_prediction(X_train, X_test, y_train, y_test, n_neighbors):
  knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
  knn.fit(X_train, y_train)
  #
  # INFO: Be sure to always keep the domain of the problem in mind! It's
  # WAY more important to errantly classify a benign tumor as malignant,
  # and have it removed, than to incorrectly leave a malignant tumor, believing
  # it to be benign, and then having the patient progress in cancer. Since the UDF
  # weights don't give you any class information, the only way to introduce this
  # data into SKLearn's KNN Classifier is by "baking" it into your data. For
  # example, randomly reducing the ratio of benign samples compared to malignant
  # samples from the training set.

  # Calculate + Print the accuracy of the testing set
  #
  score = knn.score(X_test, y_test)
  return knn, score

def process():

  X, y = clean_data_set_and_get_labels(get_data_set())

  scalers = initialize_scalers_map(X)

  score_map = dict()
  for scaler_name, T in scalers.items():
      print(scaler_name)
      #feature scaling
      if T is not None: X = T.transform(X)

      X_train, X_test, y_train, y_test = split_data_set(X, y)
      X_train, X_test = dimensionality_reduction(X_train, X_test)

      #
      # Implement and train KNeighborsClassifier on your projected 2D
      # training data here. You can use any K value from 1 - 15, so play around
      # with it and see what results you can come up. Your goal is to find a
      # good balance where you aren't too specific (low-K), nor are you too
      # general (high-K). You should also experiment with how changing the weights
      # parameter affects the results.
      #
      for i in range(1,16):
          knn, score = process_prediction(X_train, X_test, y_train, y_test, i)
          if score_map.get(scaler_name) is None: score_map[scaler_name] = []
          score_map[scaler_name].append(score)
          model_to_visualize = [6, 10]
          if i in model_to_visualize: plotDecisionBoundary(knn, X_test, y_test)

  display_scores(score_map)

def display_scores(score_map):
  print("Display Max scores per Scaler")
  for scaler_name, list_score in score_map.items():
    max_score = max(list_score)
    best_n_neigbors = list_score.index(max_score)
    print("Scaler: "+scaler_name+ " | best score: "+str(max_score) + " | n_neighbors: "+ str(best_n_neigbors))


def get_data_set():
  #
  # Load in the dataset, identify nans, and set proper headers.
  # Be sure to verify the rows line up by looking at the file in a text editor.
  #
  names= ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
  df = pd.read_csv('./Datasets/breast-cancer-wisconsin.data', header=None, names=names, na_values=['?'])
  print(df.head())
  print(df.dtypes)
  return df

def clean_data_set_and_get_labels(df):
  #
  # Copy out the status column into a slice, then drop it from the main
  # dataframe. You can also drop the sample column, since that doesn't provide
  # us with any machine learning power.
  #
  labels = df['status'].copy()
  df = df.drop(labels=['sample','status'], axis=1)

  print(df.describe())
  print(df.nuclei.unique())
  # With the labels safely extracted from the dataset, replace any nan values
  # with the mean feature / column value
  #
  # labels = labels.fillna(labels.mean())
  df = df.fillna(df.mean())
  print(df.nuclei.unique())
  return df, labels

def split_data_set(X, y):
  #
  # Do train_test_split. Use the same variable names as on the EdX platform in
  # the reading material, but set the random_state=7 for reproduceability, and keep
  # the test_size at 0.5 (50%).
  #
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7 )
  return X_train, X_test, y_train, y_test


process()