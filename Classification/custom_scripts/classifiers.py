import sklearn.metrics as sm
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from keras.models import Sequential

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csc_matrix, csr_matrix




class Classifier:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, train=False, params={}):
        if X_test is not None and y_test is not None:
            self.set_test(X_test, y_test)
        if train:
            self.fit(X_train, y_train, params)

    def __eq__(self, other):
        return (other == 'base_classifier')

    def set_test(self, X_test, y_test):
        self.X_test = csc_matrix(X_test)
        self.y_test = y_test

    def fit(self, X_train, y_train, params={}):
        raise Exception('No Classifier found. Did you implement it correctly?')

    def evaluate(self, X_test=None, y_test=None):
        assert self.clf is not None, 'Please train the classifier first!'
        assert self.X_test is not None and self.y_test is not None, 'Please set test data or provide it in function call'
        if X_test is not None and y_test is not None:
            self.set_test(X_test, y_test)

        pred = self.clf.predict(self.X_test)
        e = {}

        e['accuracy'] = sm.accuracy_score(self.y_test, pred)
        e['accuracy_balanced'] = sm.balanced_accuracy_score(self.y_test, pred)
        e['precision'] = sm.precision_score(self.y_test, pred)
        e['recall'] = sm.recall_score(self.y_test, pred)
        return e

    def confusion(self):
        assert self.clf is not None, 'Please train the classifier first!'
        assert self.X_test is not None and self.y_test is not None, 'Please set test data or provide it in function call'
        pred = self.clf.predict(self.X_test)
        cm = sm.confusion_matrix(self.y_test, pred)
        return cm



class KNN(Classifier):
    def __eq__(self, other):
        return (other == 'knn')

    def fit(self, X_train, y_train, params={}):
        self.clf = KNeighborsClassifier(**params)
        self.clf.fit(X_train, y_train)

    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

class NB(Classifier):
    def __eq__(self, other):
        return (other == 'nb')
    def fit(self, X_train, y_train, params={}):
        self.clf = MultinomialNB(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class NBB(Classifier):
    def __eq__(self, other):
        return (other == 'nbb')
    def fit(self, X_train, y_train, params={}):
        self.clf = BernoulliNB(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class DT(Classifier):
    def __eq__(self, other):
        return (other == 'dt')
    def fit(self, X_train, y_train, params={}):
        self.clf = DecisionTreeClassifier(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class RF(Classifier):
    def __eq__(self, other):
        return (other == 'rf')
    def fit(self, X_train, y_train, params={}):
        self.clf = RandomForestClassifier(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class SVM(Classifier):
    def __eq__(self, other):
        return (other == 'svm')
    def fit(self, X_train, y_train, params={}):
        self.clf = SVC(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class XGB(Classifier):
    def __eq__(self, other):
        return (other == 'xgb')
    def fit(self, X_train, y_train, params={}):
        self.clf = XGBClassifier(**params)
        self.clf.fit(csc_matrix(X_train), y_train)

class NN(Classifier):
    def __eq__(self, other):
        return (other == 'nn')

    def create_model(self, layers):
        # create model
        model = Sequential()
        for l in layers:
            model.add(l)

        model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, params={}):
        try:
            layers = params['layers']
            epochs = params['epochs']
            self.clf = self.create_model(layers)
            self.clf.fit(csc_matrix(X_train), y_train, epochs=epochs, batch_size=16, verbose=0)
        except:
            raise Exception('Parameters incorrectly formatted. Please provide a dictionary with layers and epochs')

    def evaluate(self, X_test=None, y_test=None):
        assert self.clf is not None, 'Please train the classifier first!'
        assert self.X_test is not None and self.y_test is not None, 'Please set test data or provide it in function call'
        if X_test is not None and y_test is not None:
            self.set_test(X_test, y_test)

        pred = self.clf.predict_classes(self.X_test)
        e = {}

        e['accuracy'] = sm.accuracy_score(self.y_test, pred)
        e['accuracy_balanced'] = sm.balanced_accuracy_score(self.y_test, pred)
        e['precision'] = sm.precision_score(self.y_test, pred)
        e['recall'] = sm.recall_score(self.y_test, pred)
        return e

class RuleClassifier():
    def get_label(self, row):
        score = 0
        if (row.loc[row.index[0], 'PosTags'][0].startswith('V') or 'oder' in row.loc[row.index[0], 'Feature'][-2:]) \
                and not 'oder' in row.loc[row.index[0], 'Feature'][:-2] \
                and not 'jemand' in row.loc[row.index[0], 'Feature'][:-2] \
                and not 'wer' in row.loc[row.index[0], 'Feature'] \
                and not 'was' in row.loc[row.index[0], 'Feature'] \
                and not 'welche' in row.loc[row.index[0], 'Feature'] \
                and not 'welchem' in row.loc[row.index[0], 'Feature'] \
                and not 'wieso' in row.loc[row.index[0], 'Feature'] \
                and not 'wo' in row.loc[row.index[0], 'Feature'] \
                and not 'warum' in row.loc[row.index[0], 'Feature'] \
                and not 'wie' in row.loc[row.index[0], 'Feature'] \
                and not 'denn' in row.loc[row.index[0], 'Feature'] \
                and not 'wann' in row.loc[row.index[0], 'Feature']:
            return 1
        else:
            return 0

    def predict(self, data):
        label = []
        for i in data.index.values.tolist():
            label.append(self.get_label(data.loc[[i]]))
        return (np.array(label))


    def accuracy(self, data):
        labels = self.predict(data)
        false_positive = 0
        false_negative = 0
        total_pos = data['Label'].values.tolist().count(1)
        found_pos = labels.count(1)
        for index, value in enumerate(labels):
            if value != data.loc[index, 'Label']:
                if value == 1:
                    false_positive += 1
                else:
                    false_negative += 1
                    print(index, data.loc[index, 'Feature'], value, data.loc[index, 'Label'])

        print('Accuracy:', (len(labels) - (false_positive + false_negative)) / len(labels))
        print('False Positive: ', false_positive / len(labels))
        print('False Negative: ', false_negative / len(labels))
        print('Precision: ', (found_pos - false_positive) / found_pos)
        print('Recall: ', (found_pos - false_positive) / total_pos)
        return ((len(labels) - (false_positive + false_negative)) / len(labels))

class MC(Classifier):
    def __eq__(self, other):
        return (other == 'mc')

    def fit(self, X_train=None, y_train=None, params={}):
        self.clf = RuleClassifier()

    def set_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
