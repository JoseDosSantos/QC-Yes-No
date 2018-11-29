from import_data import load_data
from feature_set_generator import FeatureSetGenerator
from pympler import asizeof
from pympler.classtracker import ClassTracker
from time import time
import util
import classifiers as c
from copy import copy
from collections import Counter

tracker = ClassTracker()
data_set = load_data()
print('Data loaded.')

trained = util.load_pickle(name='fs_1', path='..\\pickles\\feature_sets\\')
test = util.load_pickle(name='fs_test_1', path='..\\pickles\\test_features\\')

test_data = test['data_set']
featureset = 'fs_words_bigrams_pos'

X_train, y_train = trained[featureset], trained['labels']
X_test, y_test = test[featureset], test['labels']
feat_size = X_train.shape[1]

da = copy(test_data)
da['Feature'] = da['Feature'].apply(' '.join)


norm = Counter(da['Label'])[0] / Counter(da['Label'])[1]
counts = {}
for word in trained['bag_words'][1:]:
    #print(word)
    a = Counter(da[da['Feature'].str.contains(r'\b' + word + r'\b')]['Label'])
    counts[word] = a[0] / norm - a[1]
import operator
counts = sorted(counts.items(), key=operator.itemgetter(1))

l = trained['data_set']['Feature'].values.tolist()
l1 = [i for i in l if 'A' in i]
mc = c.MC(X_test=test_data, y_test=y_test)
mc.fit(X_train=X_train, y_train=y_train)

pred = mc.clf.predict(test_data)

for i, v in enumerate(pred):
    if v != y_test[i] and v == 0:
        print(i, ' '.join(test_data['Feature'][i]), 'Pred:', v, 'True:', y_test[i])

