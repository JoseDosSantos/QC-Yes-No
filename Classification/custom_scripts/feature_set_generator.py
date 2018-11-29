import pandas as pd
import numpy as np
from nltk.util import ngrams
from collections import Counter
from copy import deepcopy
import math
import pickles
from scipy.sparse import csc_matrix, hstack

class FeatureSetGenerator:
    def __init__(self, data_set=None, size=1, full_init=False, load_from_file=False, export=False):
        if load_from_file:
            self.load_from_file(size)
        elif data_set is not None:
            self.data_set = data_set.sample(frac=size)
            self.labels = np.array(self.data_set['Label'])
            if full_init:
                print('Creating feature set based on {}% of the data set. This may take a while.'.format(size*100))
                self.bag_of_words()
                self.bag_of_bigrams()
                self.bag_of_words_min()
                self.pos_tags()
                self.tfidf()
                self.word2vec()
                self.doc2vec()
                print('Created all feature sets.')
            if export:
                print('Saving to pickle.')
                self.export(size)
                print('Export finished.')

    def __getitem__(self, item):
        index = {
            'data_set': self.data_set,
            'features': self.data_set['Feature'],
            'labels': self.labels,

            'fs_words': self.fs_words,
            'fs_words_min': self.fs_words_min,
            'fs_bigrams': self.fs_bigrams,
            'fs_pos': self.fs_pos,
            'fs_words_pos': self.combine([self.fs_words, self.fs_pos]),
            'fs_words_min_pos': self.combine([self.fs_words_min, self.fs_pos]),
            'fs_bigrams_pos': self.combine([self.fs_bigrams, self.fs_pos]),
            'fs_words_bigrams': self.combine([self.fs_words, self.fs_bigrams]),
            'fs_words_bigrams_pos': self.combine([self.fs_words, self.fs_bigrams, self.fs_pos]),
            'fs_words_min_bigrams_pos': self.combine([self.fs_words_min, self.fs_bigrams, self.fs_pos]),
            'fs_w2v': self.fs_w2v,
            'fs_d2v': self.fs_d2v,
            'fs_tfidf': self.fs_tfidf,

            'bag_words': self.bag_words,
            'bag_words_min': self.bag_words_min,
            'bag_bigrams': self.bag_bigrams,
            'bag_pos': self.bag_pos,
            'bag_w2v': self.bag_w2v
        }
        try:
            return index[item]
        except:
            return None

    def set_data(self, data_set, size=1):
        self.data_set = data_set.sample(frac=size)

    def load_data(self, file_path, size=1):
        try:
            data_set = pd.read_csv(file_path, sep=';').drop('Unnamed: 0', 1)
            data_set['Feature'] = data_set['Feature'].apply(lambda x: x.split('#'))
            data_set['PosTags'] = data_set['PosTags'].apply(lambda x: x.split('#'))
            self.data_set = data_set
        except Exception as e:
            print('Could not load file. Error:{}'.format(e))


    def bag_of_words(self):
        self.bag_words = sorted(set(word for row in self.data_set['Feature'] for word in row))
        print('Created bag of words. Length: {0}'.format(len(self.bag_words)))
        self.fs_words = csc_matrix([[word in getattr(row, 'Feature') for word in self.bag_words] for row in self.data_set.itertuples()])
        print('Created Featureset "{0}" for {1} documents.'.format('words', self.fs_words.shape[0]))

    def encode_fs_words(self, data_set):
        fs_words = csc_matrix([[word in getattr(row, 'Feature') for word in self.bag_words] for row in data_set.itertuples()])
        return fs_words, self.bag_words


    def bag_of_bigrams(self):
        self.bag_bigrams = sorted(set(gram for row in self.data_set['Feature'] for gram in ngrams(row, 2)))
        print('Created bag of bigrams. Length: {0}'.format(len(self.bag_bigrams)))
        self.fs_bigrams = csc_matrix([[gram in ngrams(getattr(row, 'Feature'), 2) for gram in self.bag_bigrams] for row in self.data_set.itertuples()])
        print('Created Featureset "{0}" for {1} documents.'.format('bigrams', self.fs_bigrams.shape[0]))


    def encode_fs_bigrams(self, data_set):
        fs_bigrams = csc_matrix([[gram in ngrams(getattr(row, 'Feature'), 2) for gram in self.bag_bigrams] for row in data_set.itertuples()])
        return fs_bigrams, self.bag_bigrams


    def bag_of_words_min(self, min_count=5):
        assert self.bag_words is not None, "Please build bag of words first!"

        self.min_word_count = min_count

        sents = self.data_set['Feature'].values.tolist()
        sent_merged = [j for i in sents for j in i]
        self.word_count = Counter(sent_merged)

        self.bag_words_min = sorted(set(word for word in self.bag_words if self.word_count[word] >= self.min_word_count))
        print('Created bag of words with minimum count {0}. Length: {1}'.format(self.min_word_count, len(self.bag_words_min)))

        min_word_frame = deepcopy(self.data_set)
        min_word_frame['Feature_min'] = ''

        for index, row in min_word_frame.iterrows():
            f = []
            for word in row['Feature']:
                if self.word_count[word] >= self.min_word_count:
                    f.append(word)
            min_word_frame.at[index, 'Feature_min'] = f

        self.fs_words_min = csc_matrix([[word in getattr(row, 'Feature_min') for word in self.bag_words_min] for row in min_word_frame.itertuples()])
        print('Created Featureset "{0}" for {1} documents.'.format('words min', self.fs_words_min.shape[0]))

    def encode_fs_words_min(self, data_set):
        min_word_frame = deepcopy(data_set)
        min_word_frame['Feature_min'] = ''

        for index, row in min_word_frame.iterrows():
            f = []
            for word in row['Feature']:
                if self.word_count[word] >= self.min_word_count:
                    f.append(word)
            min_word_frame.at[index, 'Feature_min'] = f
        fs_words_min = csc_matrix([[word in getattr(row, 'Feature_min') for word in self.bag_words_min] for row in min_word_frame.itertuples()])
        return fs_words_min, self.bag_words_min


    def pos_tags(self):
        self.bag_pos = sorted(set(tag for row in self.data_set['PosTags'] for tag in row))
        print('Created bag of POS-tags. Length: {0}'.format(len(self.bag_pos)))
        self.fs_pos = csc_matrix([[tag in getattr(row, 'PosTags') for tag in self.bag_pos] for row in self.data_set.itertuples()])
        print('Created Featureset "{0}" for {1} documents.'.format('pos', self.fs_pos.shape[0]))


    def encode_fs_pos(self, data_set):
        fs_pos = csc_matrix([[tag in getattr(row, 'PosTags') for tag in self.bag_pos] for row in data_set.itertuples()])
        return fs_pos, self.bag_pos


    def tfidf(self):
        assert self.bag_words is not None and self.fs_words is not None, "Please build bag of words and fs words first!"
        from textblob import TextBlob as tb
        def tf(word, blob):
            return blob.words.count(word) / len(blob.words)

        def n_containing(word, bloblist):
            return sum(1 for blob in bloblist if word in blob.words)

        def idf(word, bloblist):
            return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

        def tfidf(word, blob, bloblist):
            return tf(word, blob) * idf(word, bloblist)

        tfidf_frame = pd.DataFrame(index=range(len(self.data_set)), columns=self.bag_words).fillna(0.0)
        data_set_sent = [' '.join(data['Feature']) for _, data in self.data_set.iterrows()]

        self.bloblist = [tb(i) for i in data_set_sent]
        tfidf_list = []
        for i, blob in enumerate(self.bloblist):
            scores = {word: tfidf(word, blob, self.bloblist) for word in blob.words}
            for word in scores:
                tfidf_frame.at[i, word] = round(scores[word], 5)
            tfidf_list.append(scores)

        self.fs_tfidf = csc_matrix(tfidf_frame)
        self.data_set['TfIdf'] = tfidf_list
        print('Created Featureset "{0}" for {1} documents.'.format('tfidf', self.fs_tfidf.shape[0]))


    def encode_tfidf(self, data_set):
        ds = data_set
        from textblob import TextBlob as tb
        def tf(word, blob):
            return blob.words.count(word) / len(blob.words)

        def n_containing(word):
            return sum(1 for blob in self.bloblist if word in blob.words)

        def idf(word):
            return math.log(len(self.bloblist) / (1 + n_containing(word)))

        def tfidf(word, blob):
            return tf(word, blob) * idf(word)

        tfidf_frame = pd.DataFrame(index=range(len(ds)), columns=self.bag_words).fillna(0.0)
        data_set_sent = [' '.join(data['Feature']) for _, data in ds.iterrows()]
        bloblist = [tb(i) for i in data_set_sent]
        tfidf_list = []

        for i, blob in enumerate(bloblist):
            scores = {word: tfidf(word, blob) for word in blob.words}
            for word in scores:
                try:
                    tfidf_frame.at[i, word] = round(scores[word], 5)
                except:
                    tfidf_frame.at[i, word] = 0
            tfidf_list.append(scores)

        fs_tfidf = csc_matrix(tfidf_frame)
        ds['TfIdf'] = tfidf_list
        return ds, fs_tfidf


    def word2vec(self):
        assert self.bag_words is not None and self.fs_words is not None, "Please build bag of words and fs words first!"
        assert 'TfIdf' in self.data_set.columns, 'No TfIdf Vectors found. Please calculate Tfidf Vectors first!'

        import gensim
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('..\\german.model', binary=True, )
        print('Loaded word2vec model.')

        vectors = []

        def get_w2v(word):
            try:
                vec = w2v_model.get_vector(word)
                return np.round(vec, 8)
            except:
                vec = np.zeros(shape=(1, 300))
                return vec[0]

        self.bag_w2v = {}
        for word in self.bag_words:
            self.bag_w2v[word] = get_w2v(word)
        print('Created bag of Word2Vec Vectors. Length:{0}'.format(len(self.bag_w2v)))


        for i, row in enumerate(self.data_set.itertuples()):
            avg_vecs = []
            word_len = 0
            for j, word in enumerate(getattr(row, 'Feature')[:-1]):
                vec = self.bag_w2v[word]
                if not sum(vec) == 0:
                    word_len += 1

                avg_vecs.append(list(vec * getattr(row, 'TfIdf')[word]))

            if word_len > 0:
                vectors.append(np.array(avg_vecs).sum(axis=0) / word_len)
            else:
                vectors.append(np.array(avg_vecs).sum(axis=0))

        self.fs_w2v = csc_matrix(np.array(vectors).round(6))
        print('Created Featureset "{0}" for {1} documents.'.format('word2vec', self.fs_w2v.shape[0]))


    def encode_word2vec(self, data_set, bag_words, use_model=False, model_path='..\\german.model'):
        vectors = []

        def get_w2v(word):
            try:
                vec = w2v_model.get_vector(word)
                return np.round(vec, 8)
            except:
                vec = np.zeros(shape=(1, 300))
                return vec[0]

        if use_model:
            import gensim
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True, )
            bag_w2v = {}
            for word in bag_words:
                bag_w2v[word] = get_w2v(word)
        else:
            bag_w2v = self.bag_w2v

        for i, row in enumerate(data_set.itertuples()):
            avg_vecs = []
            word_len = 0
            for j, word in enumerate(getattr(row, 'Feature')[:-1]):
                try:
                    vec = bag_w2v[word]
                except:
                    vec = np.zeros(shape=(1, 300))
                if not sum(vec) == 0:
                    word_len += 1

                avg_vecs.append(list(vec * getattr(row, 'TfIdf')[word]))

            if word_len > 0:
                vectors.append(np.array(avg_vecs).sum(axis=0) / word_len)
            else:
                vectors.append(np.array(avg_vecs).sum(axis=0))

        fs_w2v = csc_matrix(np.array(vectors).round(6))
        return fs_w2v


    def doc2vec(self, return_model=False):
        import gensim
        def read_corpus(data):
            for row in data.itertuples():
                yield gensim.models.doc2vec.TaggedDocument(getattr(row, 'Feature'), tags=[getattr(row, 'Label')])

        shuffled = self.data_set.sample(frac=1)
        corpus = list(read_corpus(shuffled))

        self.model_doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=40)
        self.model_doc2vec.build_vocab(corpus)
        self.model_doc2vec.train(corpus, total_examples=self.model_doc2vec.corpus_count, epochs=self.model_doc2vec.epochs)
        self.model_doc2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        print('Successfully trained Doc2Vec model on {0} documents.'.format(len(corpus)))
        vectors = []

        for row in self.data_set.itertuples():
            vectors.append(self.model_doc2vec.infer_vector(getattr(row, 'Feature')))
        self.fs_d2v = csc_matrix(np.array(vectors).round(6))
        print('Created Featureset "{0}" for {1} documents.'.format('doc2vec', self.fs_d2v.shape[0]))

    def encode_doc2vec(self, data_set):
        vectors = []
        for row in data_set.itertuples():
            vectors.append(self.model_doc2vec.infer_vector(getattr(row, 'Feature')))
        fs_d2v = csc_matrix(np.array(vectors).round(6))
        return fs_d2v

    def combine(self, sets):
        return hstack(sets)

    def export(self, size):
        pickles.create_pickle(self, '\\feature_sets\\fs_' + str(size))

    def load_from_file(self, size):
        self = pickles.load_pickle('\\feature_sets\\fs_' + str(size))

    def get_stats(self):
        stats = {}
        stats['Size'] = len(self.data_set)
        stats['Labels'] = Counter(self.data_set['Label'])
        stats['Words'] = len(self.bag_words)
        stats['Words_min_count_' + str(self.min_word_count)] = len(self.bag_words_min)
        stats['Bigrams'] = len(self.bag_bigrams)
        stats['Pos_tags'] = len(self.bag_pos)
        return stats

