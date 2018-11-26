import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from custom_scripts.import_data import load_data
from custom_scripts.util import load_pickle, get_train_test
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from keras.constraints import maxnorm
from keras.optimizers import SGD


#data_set = load_data()
#print('Data loaded.')

#fs_words = load_pickle('featuresets/fs_words')
#fs_ngrams = load_pickle('featuresets/fs_ngrams')
#fs_pos = load_pickle('featuresets/fs_pos')
#fs_words_min = load_pickle('featuresets/fs_words_min')
#fs_ngrams_min = load_pickle('featuresets/fs_ngrams_min')
#fs_tfidf_words = load_pickle('featuresets/fs_tfidf_words')
#fs_words_ngrams = load_pickle('featuresets/fs_words_ngrams')
#fs_words_pos = load_pickle('featuresets/fs_words_pos')
#fs_words_min_pos = load_pickle('featuresets/fs_min_pos')
#fs_ngrams_pos = load_pickle('featuresets/fs_ngrams_pos')
fs_words_ngrams_pos = load_pickle('featuresets/fs_words_ngrams_pos')
#fs_w2v = load_pickle('featuresets/fs_w2v')
#fs_d2v = load_pickle('featuresets/fs_d2v')
print('Models loaded.')
seed = 1
np.random.seed(seed)
"""
model = keras.Sequential()
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(500, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(3)))
#model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#opt = keras.optimizers.SGD(lr=0.01, nesterov=True)
opt = tf.train.AdamOptimizer()
#loss = 'sparse_categorical_crossentropy'
loss = 'binary_crossentropy'
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
"""

def create_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.5, input_shape=(24686,)))
    model.add(Dense(500, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    # sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    opt = tf.train.AdamOptimizer()
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = get_train_test(fs_words_ngrams_pos, test_size=0, random_state=1)

np.random.seed(seed)
estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=10, batch_size=16, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#print('Starting training.')
#print(test_loss, test_acc)
