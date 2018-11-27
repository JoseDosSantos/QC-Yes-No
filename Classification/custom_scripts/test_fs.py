import pickles

class TestFS:
    def __init__(self, data_set=None):
        if data_set is not None:
            self.data_set = data_set

    def generate_features(self, fs, data_set=None):
        if data_set is not None:
            self.data_set = data_set

        self.fs_words = fs.encode_fs_words(self.data_set)
        print('words', len(self.fs_words), len(self.fs_words['Feature'][0]))
        self.fs_bigrams = fs.encode_fs_bigrams(self.data_set)
        print('bigrams', len(self.fs_bigrams), len(self.fs_bigrams['Feature'][0]))
        self.fs_words_min = fs.encode_fs_words_min(self.data_set)
        print('words_min', len(self.fs_words_min), len(self.fs_words_min['Feature'][0]))
        self.fs_pos = fs.encode_fs_pos(self.data_set)
        print('pos', len(self.fs_pos), len(self.fs_pos['Feature'][0]))
        self.data_set, self.fs_tfidf = fs.encode_tfidf(self.data_set)
        print('tfidf', len(self.fs_tfidf), len(self.fs_tfidf['Feature'][0]))
        self.fs_w2v = fs.encode_word2vec(self.data_set, set(word for row in self.data_set['Feature'] for word in row), use_model=True)
        print('w2v', len(self.fs_w2v), len(self.fs_w2v['Feature'][0]))
        self.fs_d2v = fs.encode_doc2vec(self.data_set)
        print('d2v', len(self.fs_d2v), len(self.fs_d2v ['Feature'][0]))
        self.fs_words_bigrams = fs.combine([self.fs_words, self.fs_bigrams])
        print('words_bigrams', len(self.fs_words_bigrams), len(self.fs_words_bigrams['Feature'][0]))
        self.fs_words_pos = fs.combine([self.fs_words, self.fs_pos])
        print('words_pos', len(self.fs_words_pos), len(self.fs_words_pos['Feature'][0]))
        self.fs_words_min_pos = fs.combine([self.fs_words_min, self.fs_pos])
        print('words_min_pos', len(self.fs_words_min_pos), len(self.fs_words_min_pos['Feature'][0]))
        self.fs_words_bigrams_pos = fs.combine([self.fs_words, self.fs_bigrams, self.fs_pos])
        print('words_bigrams_pos', len(self.fs_words_bigrams_pos), len(self.fs_words_bigrams_pos['Feature'][0]))
        self.fs_words_min_bigrams_pos = fs.combine([self.fs_words_min, self.fs_bigrams, self.fs_pos])
        print('words_min_bigrams_pos', len(self.fs_words_min_bigrams_pos), len(self.fs_words_min_bigrams_pos['Feature'][0]))

    def export(self, path, name):
        pickles.create_pickle(self, str(name), path)

