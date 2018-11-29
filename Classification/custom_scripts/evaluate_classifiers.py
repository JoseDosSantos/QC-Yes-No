from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import ParameterGrid
from time import time

import numpy as np
import pandas as pd
import sys
sys.path.append('..')

import util
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='imp')
import classifiers


def grid_search(X_train, y_train, clf, params):
    grid = list(ParameterGrid(params))
    scores = {'best': grid[0],
              'accuracy': 0,
              'precision': 0,
              'recall': 0,
              'time': 0}
    for p in grid:
        s = time()
        try:
            clf.fit(X_train, y_train, p)
            e = clf.evaluate()
            if clf == 'nn':
                p['layers'] = [l.get_config() for l in p['layers']]

            if e['accuracy'] > scores['accuracy']:
                t = time() - s
                scores['accuracy'] = e['accuracy']
                scores['precision'] = e['precision']
                scores['recall'] = e['recall']
                scores['best'] = p
                scores['time'] = t
        except Exception as e:
            print('An error occured.')
            print(e)
            continue


    print(scores, p)
    return scores

def eval_classifier(data_set, fsplit, featureset, X_train, y_train, X_test, y_test):
    feat_size = len(X_train[0])
    print(fsplit, featureset)
    #clf = classifiers.MC(X_test=data_set, y_test=y_test)
    #clf.fit()
    #print(clf.evaluate())

    params = {'knn': {'classifier': classifiers.KNN,
                      'params': {'n_neighbors': [5, 15, 55],
                                 'leaf_size': [10, 30, 100],
                                 'p': [2, 3],
                                 'n_jobs': [-1]}},
              'nb': {'classifier': classifiers.NB,
                     'params': {'alpha': [0.5, 1.0, 1.5]}},
              'dt': {'classifier': classifiers.DT,
                     'params': {'max_depth': [None, 5, 8],
                                'min_samples_leaf': [1, 2, 3]}},
              'rf': {'classifier': classifiers.RF,
                     'params': {'n_estimators': [50, 100, 500],
                                'min_samples_leaf': [2, 10, 20],
                                'n_jobs': [-1]}},
              'svm': {'classifier': classifiers.SVM,
                      'params': {'kernel': ['linear'],
                                 'C': [0.5, 0.75, 1, 2]}},
              'xgb': {'classifier': classifiers.XGB,
                      'params': {'n_estimators': [50, 100, 400],
                                 'max_depth': [3, 5, 8, 10],
                                 'learning_rate': [0.075, 0.1, 0.125]}},
              'nn': {'classifier': classifiers.NN,
                     'params': {'layers': [[Dropout(0.5, input_shape=(feat_size,)),
                                            Dense(500, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)),
                                            Dropout(0.5),
                                            Dense(1, kernel_initializer='normal', activation='sigmoid')],
                                           [Dropout(0.5, input_shape=(feat_size,)),
                                            Dense(50, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)),
                                            Dropout(0.5),
                                            Dense(1, kernel_initializer='normal', activation='sigmoid')],
                                           [Dropout(0.5, input_shape=(feat_size,)),
                                            Dense(50, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)),
                                            Dropout(0.5),
                                            Dense(50, kernel_initializer='normal', activation='sigmoid'),
                                            Dropout(0.25),
                                            Dense(1, kernel_initializer='normal', activation='sigmoid')]],
                                'epochs': [10]}}
              }


    scores = {}

    for c in params:
        print(c)
        if c == 'nb' and (featureset == 'fs_w2v' or featureset == 'fs_d2v'):
            print('Skipped due to negative values')
            continue
        clf = params[c]['classifier'](X_test=X_test, y_test=y_test, train=False)
        scores[c] = grid_search(X_train, y_train, clf, params=params[c]['params'])

    out = pd.DataFrame(scores)
    out.to_csv('stats\\' + fsplit + '\\' + featureset + '.csv', sep=';')


def run():
    feature_splits = [1]#, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    sets = ['fs_words', 'fs_words_min', 'fs_bigrams', 'fs_pos', 'fs_words_pos', 'fs_words_min_pos', 'fs_bigrams_pos',
            'fs_words_bigrams', 'fs_words_bigrams_pos', 'fs_words_min_bigrams_pos', 'fs_w2v', 'fs_d2v']  # , 'fs_tfidf']

    for fsplit in feature_splits:
        test = util.load_pickle(name='fs_test_' + str(fsplit), path='..\\pickles\\test_features\\')
        trained = util.load_pickle(name='fs_' + str(fsplit), path='..\\pickles\\feature_sets\\')
        print('Loaded:', str(fsplit), 'with feature fs_words')

        for s in sets:
            featureset = s
            test_set = test[featureset]
            trained_set = trained[featureset]
            X_train, y_train = np.array(list(trained_set['Feature'])), np.array(trained_set['Label'])
            X_test, y_test = np.array(list(test_set['Feature'])), np.array(test_set['Label'])

            eval_classifier(data_set=test['data_set'], fsplit=str(fsplit), featureset=featureset, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    print(1)
    run()





