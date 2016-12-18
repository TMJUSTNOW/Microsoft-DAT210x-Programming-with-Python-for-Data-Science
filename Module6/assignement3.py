from collections import OrderedDict
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.cross_validation import train_test_split



class ModelParams:
    c = None
    gamma = None
    scaler_name = None
    def __init__(self, c, gamma, scaler_name):
        self.gamma = gamma
        self.c = c
        self.scaler_name = scaler_name


def display_scores(scores):
    sorterd_scores = OrderedDict(sorted(scores.items(), key=itemgetter(1)))
    print("##### Display scores #####")
    for params, score in sorterd_scores.items():
        print("score :", score, " | gamma: ", params.gamma, " | c: ", params.c, " | scaler_name: ", params.scaler_name)


def initialize_scalers_map(X):

  scalers = dict()
  #scalers['NoScaler'] = None
  scalers['Normalizer'] = preprocessing.Normalizer().fit(X)
  scalers['MaxAbsScaler'] = preprocessing.MaxAbsScaler().fit(X)
  scalers['MinMaxScaler'] = preprocessing.MinMaxScaler().fit(X)
  scalers['KernelCenterer'] = preprocessing.KernelCenterer().fit(X)
  scalers['StandardScaler'] = preprocessing.StandardScaler().fit(X)

  return scalers

X = pd.read_csv('./Datasets/parkinsons.data')
print(X.describe())
# INFO: An easy way to show which rows have nans in them
print(X[pd.isnull(X).any(axis=1)])

y = X['status'].copy()

X = X.drop(['name', 'status'], axis =1)
print(y.head())

svc = svm.SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7 )


C_range = np.arange(0.05,2,0.05) # python range() does not allowed float
gamma_range = np.arange(0.001, 0.1, 0.001)

best_score = 0
scores = dict() # map<params , score>
scalers = initialize_scalers_map(X_train)
for scaler_name, T in scalers.items():
    X_scale_train = T.transform(X_train)
    X_scale_test = T.transform(X_test)
    for c in C_range:
        for gamma in gamma_range:
            params = ModelParams(c, gamma, scaler_name)
            svc = svm.SVC(C=c, gamma=gamma)
            svc.fit(X_scale_train, y_train)
            score = svc.score(X_scale_test , y_test)
            if score > best_score: best_score = score
            scores[params] = score

print("Meilleur score", best_score)
display_scores(scores)

