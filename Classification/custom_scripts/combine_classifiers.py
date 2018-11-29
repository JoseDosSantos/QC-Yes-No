import classifiers as c
from sklearn.ensemble import VotingClassifier
import util
import pickles
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from scipy.sparse import csc_matrix
from pympler import asizeof
from pympler.classtracker import ClassTracker

def size(data):
    return asizeof.asizeof(data)


def export_classifiers():
    trained = util.load_pickle(name='fs_1', path='..\\pickles\\feature_sets\\')
    print('trained', size(trained))
    test = util.load_pickle(name='fs_test_1', path='..\\pickles\\test_features\\')
    print('test', size(test))

    test_data = test['data_set']
    featureset = 'fs_words_bigrams_pos'

    X_train, y_train = trained[featureset], trained['labels']
    X_test, y_test = test[featureset], test['labels']
    feat_size = X_train.shape[1]

    knn = c.KNN(X_test=X_test.toarray(), y_test=y_test)
    nb = c.NB(X_test=X_test, y_test=y_test)
    dt = c.DT(X_test=X_test, y_test=y_test)
    rf = c.RF(X_test=X_test, y_test=y_test)
    xgb = c.XGB(X_test=X_test, y_test=y_test)
    svm = c.SVM(X_test=X_test, y_test=y_test)
    nn = c.NN(X_test=X_test, y_test=y_test)
    mc = c.MC(X_test=test_data, y_test=y_test)


    knn.fit(X_train.toarray(), y_train, params={'leaf_size': 100, 'n_jobs': -1, 'n_neighbors': 55, 'p': 3})
    nb.fit(X_train, y_train, params={'alpha': 1.5})
    dt.fit(X_train, y_train, params={'max_depth': 8, 'min_samples_leaf': 3})
    rf.fit(X_train, y_train, params={'min_samples_leaf': 20, 'n_estimators': 500, 'n_jobs': -1})
    xgb.fit(X_train, y_train, params={'learning_rate': 0.125, 'max_depth': 10, 'n_estimators': 400})
    svm.fit(X_train, y_train, params={'C': 2, 'kernel': 'linear', 'probability': True})
    nn.fit(X_train, y_train, params={'epochs': 10, 'layers': [Dropout(0.5, input_shape=(feat_size,)), Dense(50, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)),
                                                              Dropout(0.5), Dense(50, kernel_initializer='normal', activation='sigmoid'),
                                                              Dropout(0.25), Dense(1, kernel_initializer='normal', activation='sigmoid')]})

    mc.fit(X_train=X_train, y_train=y_train)
    clfs = {
        'knn': knn,
        'nb': nb,
        'dt': dt,
        'rf': rf,
        'xgb': xgb,
        'svm': svm,
        'nn': nn,
        'mc': mc
    }
    return clfs

def export(data):
    pickles.create_pickle(data, 'clfs', path='')

def load_from_file():
    data = pickles.load_pickle('clfs', path='')
    return data

def predict_from_multiple_estimator(estimators, X_list, weights = None):

    # Predict 'soft' voting with probabilities

    pred1 = np.asarray([clf.predict(csc_matrix(X_list)) for clf in estimators])
    pred2 = np.average(pred1, axis=0, weights=weights)
    pred = np.argmax(pred2, axis=1)

    # Convert integer predictions to original labels:
    return pred


#x = export_classifiers()
#print('x', size(x))
#export(x)

trained = util.load_pickle(name='fs_1', path='..\\pickles\\feature_sets\\')
print('trained', size(trained))
test = util.load_pickle(name='fs_test_1', path='..\\pickles\\test_features\\')
print('test', size(test))

test_data = test['data_set']
featureset = 'fs_words_bigrams_pos'

X_train, y_train = trained[featureset], trained['labels']
X_test, y_test = test[featureset], test['labels']
feat_size = X_train.shape[1]
x = load_from_file()
svm = x['svm']
xgb = x['xgb']
knn = x['knn']
nb = x['nb']
dt = x['dt']
rf = x['rf']
nn = x['nn']
mc = x['mc']
estimators = [svm.clf, xgb.clf,  nb.clf, dt.clf, rf.clf]#, mc.clf], #, nn.clf]
#y_pred = predict_from_multiple_estimator(estimators, X_test)

from mlxtend.classifier import EnsembleVoteClassifier

combined =EnsembleVoteClassifier(clfs=estimators, voting='hard', refit=False)
combined.fit(X_train, y_train)
y_pred = combined.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
