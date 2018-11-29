import pickles
from scipy.sparse import hstack
import numpy as np

class TestFS:
    def __init__(self, data_set=None):
        if data_set is not None:
            self.data_set = data_set

    def __getitem__(self, item):
        index = {
            'data_set': self.data_set,
            'features': self.data_set['Feature'],
            'labels': np.array(self.data_set['Label']),
            'fs_words': self.fs_words,
            'fs_words_min': self.fs_words_min,
            'fs_bigrams': self.fs_bigrams,
            'fs_pos': self.fs_pos,
            'fs_words_pos': self.fs_words_pos,
            'fs_words_min_pos': self.fs_words_min_pos,
            'fs_bigrams_pos': self.fs_words_bigrams_pos,
            'fs_words_bigrams': self.fs_words_bigrams,
            'fs_words_bigrams_pos': self.fs_words_bigrams_pos,
            'fs_words_min_bigrams_pos': self.fs_words_min_bigrams_pos,
            'fs_w2v': self.fs_w2v,
            'fs_d2v': self.fs_d2v,
            'fs_tfidf': self.fs_tfidf,
        }
        try:
            return index[item]
        except:
            return None

    def generate_features(self, fs, data_set=None):
        if data_set is not None:
            self.data_set = data_set

        self.fs_words, _ = fs.encode_fs_words(self.data_set)
        self.fs_bigrams, _ = fs.encode_fs_bigrams(self.data_set)
        self.fs_words_min, _ = fs.encode_fs_words_min(self.data_set)
        self.fs_pos, _ = fs.encode_fs_pos(self.data_set)

        self.data_set, self.fs_tfidf = fs.encode_tfidf(self.data_set)

        self.fs_w2v = fs.encode_word2vec(self.data_set, set(word for row in self.data_set['Feature'] for word in row), use_model=True)
        self.fs_d2v = fs.encode_doc2vec(self.data_set)

        self.fs_words_bigrams = self.combine([self.fs_words, self.fs_bigrams])
        self.fs_words_pos = self.combine([self.fs_words, self.fs_pos])
        self.fs_words_min_pos = self.combine([self.fs_words_min, self.fs_pos])
        self.fs_words_bigrams_pos = self.combine([self.fs_words, self.fs_bigrams, self.fs_pos])
        self.fs_words_min_bigrams_pos = self.combine([self.fs_words_min, self.fs_bigrams, self.fs_pos])

    def combine(self, sets):
        return hstack(sets)

    def export(self, path, name):
        pickles.create_pickle(self, str(name), path)

