from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def k_nearest(X_train, X_test, y_train, y_test, parameters={}):
    kn_clf = KNeighborsClassifier(**parameters)
    kn_clf.fit(X_train, y_train)
    kn_accuracy = kn_clf.score(X_test, y_test)
    return kn_accuracy

def naive_bayes(X_train, X_test, y_train, y_test, parameters = {}):
    nb_clf = MultinomialNB(**parameters)
    nb_clf.fit(X_train, y_train)
    nb_accuracy = nb_clf.score(X_test, y_test)
    return nb_accuracy

def naive_bayes_C(X_train, X_test, y_train, y_test, parameters = {}):
    nb_clf = ComplementNB(**parameters)
    nb_clf.fit(X_train, y_train)
    nb_accuracy = nb_clf.score(X_test, y_test)
    return nb_accuracy

def naive_bayes_B(X_train, X_test, y_train, y_test, parameters = {}):
    nb_clf = BernoulliNB(**parameters)
    nb_clf.fit(X_train, y_train)
    nb_accuracy = nb_clf.score(X_test, y_test)
    return nb_accuracy

def decision_tree(X_train, X_test, y_train, y_test, parameters = {}):
    dt_clf = DecisionTreeClassifier(**parameters)
    dt_clf.fit(X_train, y_train)
    dt_accuracy = dt_clf.score(X_test, y_test)
    return dt_accuracy

def random_forest(X_train, X_test, y_train, y_test, parameters = {}):
    rf_clf = RandomForestClassifier(**parameters)
    rf_clf.fit(X_train, y_train)
    rf_accuracy = rf_clf.score(X_test, y_test)
    return rf_accuracy

def SVM(X_train, X_test, y_train, y_test, parameters = {}):
    svm_clf = SVC(**parameters)
    svm_clf.fit(X_train, y_train)
    svm_accuracy = svm_clf.score(X_test, y_test)
    return svm_accuracy

def XG_Boost(X_train, X_test, y_train, y_test, parameters = {}):
    xg_clf = XGBClassifier(**parameters)
    xg_clf.fit(X_train, y_train)
    xg_accuracy = xg_clf.score(X_test, y_test)
    return xg_accuracy