import pandas as pd
import numpy as np
from pickles import create_pickle, load_pickle
from time import time
from copy import deepcopy
from sklearn.model_selection import train_test_split, GridSearchCV


def get_train_test(feature_set, test_size=0.2, f_col='Feature', l_col='Label', random_state=None):
    X, y = pd.DataFrame(feature_set[f_col]), pd.DataFrame(feature_set[l_col])
    X = np.array(list(X[f_col]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



def test_all(featureset, runs=5, test_size=0.2, f_col='Feature'):
    algs = {'k_nearest' : {'n_jobs' : -1},
             'naive_bayes': {},
             'random_forest': {'n_estimators': 100, 'n_jobs': -1, 'min_samples_split': 10},
             'decision_tree': {},
             'SVM': {'kernel' : 'linear'},
             'XG_Boost' : {'max_depth' : 5, 'n_estimators' : 125, 'learning_rate' : 0.1, 'min_child_weight' : 1, 'njobs' : -1},
             'MLP': {'solver': 'lbfgs', 'alpha': 1e-5, 'hidden_layer_sizes': (50, 2), 'random_state': 1}}

    for alg in algs:
        get_average_accuracy(alg, runs, featureset, test_size, algs[alg], f_col=f_col)


def get_average_accuracy(function_name, iteration_amount, data_set, test_size, parameters, f_col='Feature', l_col='Label'):
    results = []
    start_time = time()

    for i in range(iteration_amount):
        X_train, X_test, y_train, y_test = get_train_test(data_set, test_size, f_col, l_col)
        results.append(globals()[function_name](X_train, X_test, y_train, y_test, parameters))

        print("'Evaluated {0} questions with {1}. Average accuracy: {2}. Time taken: {3}".format(i + 1, function_name, sum(results) / (i + 1), time() - start_time), end='\r')
    print('\n')
    return sum(results) / iteration_amount


