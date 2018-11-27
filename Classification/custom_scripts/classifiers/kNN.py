from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from scipy.sparse import csc_matrix
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix

class kNN:
    def __init__(self, X_train, y_train, params={'kernel': 'linear'}):
        self.train_clf(X_train, y_train, params)

    def train_clf(self, X_train, y_train, parameters={}):
        print(len(X_train), len(y_train))
        self.clf = SVC(**parameters)
        self.clf.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        assert self.clf is not None, 'Please train the classifier first!'
        print(self.clf.score(X_test, y_test))



def accuracy(train, test, params):
    pass


import sys
sys.path.append('..')

import util

fs = util.load_pickle(name='fs_0.2', path='..\\..\\pickles\\feature_sets\\').fs_words
test = util.load_pickle(name='fs_test_0.2', path='..\\..\\pickles\\test_features\\').fs_words
#for i, x in enumerate(list(fs['Feature'])):
#    for j, y in enumerate(list(test['Feature'])):
#        if ''.join(x) == ''.join(y):
#            print(i, x, j, y)
from collections import Counter

#print(len(fs['Feature'][0]), len(test['Feature'][0]))
#print(Counter(test['Feature'][0]))
x = kNN(np.array(list(fs['Feature'])), np.array(fs['Label']))
x.evaluate(np.array(list(test['Feature'])), np.array(test['Label']))

