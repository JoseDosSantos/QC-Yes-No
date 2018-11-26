import pandas as pd
import numpy as np
from nltk.util import ngrams
from collections import Counter
from copy import deepcopy
import math
import pickles

class FeatureSetGenerator:
    def __init__(self, data_set=None, size=1, full_init=False, load_from_file=False, export=False):
        if load_from_file:
            self.load_from_file(size)
        elif data_set is not None:
            self.data_set = data_set.sample(frac=size)
            if full_init:
                print('Creating feature set based on {}% of the data set. This may take a while.'.format(size*100))
                self.bag_words, self.fs_words = self.bag_of_words()
                self.bag_bigrams, self.fs_bigrams = self.bag_of_bigrams()
                self.bag_words_min, self.fs_words_min = self.bag_of_words_min()
                self.bag_pos, self.fs_pos = self.pos_tags()
                self.fs_tfidf = self.tfidf()
                self.fs_w2v = self.word2vec()
                self.fs_d2v = self.doc2vec()

                print('Creating combined featuresets...')
                self.fs_words_bigrams = self.combine([self.fs_words, self.fs_bigrams])
                self.fs_words_pos = self.combine([self.fs_words, self.fs_pos])
                self.fs_words_min_pos = self.combine([self.fs_words_min, self.fs_pos])
                self.fs_words_bigrams_pos = self.combine([self.fs_words, self.fs_bigrams, self.fs_pos])
                self.fs_words_min_bigrams_pos = self.combine([self.fs_words_min, self.fs_bigrams, self.fs_pos])
                print('Created all feature sets.')
            if export:
                print('Saving to pickle.')
                self.export(size)
                print('Export finished.')


    def set_data(self, data_set, size=1):
        self.data_set = data_set.sample(frac=size)

    def load_data(self, file_path, size=1):
        try:
            data_set = pd.read_csv(file_path, sep=';').drop('Unnamed: 0', 1)
        except Exception as e:
            print('Could not load file. Error:{}'.format(e))
        data_set['Feature'] = data_set['Feature'].apply(lambda x: x.split('#'))
        data_set['PosTags'] = data_set['PosTags'].apply(lambda x: x.split('#'))
        self.data_set = data_set


    def name(self, variable):
        return [k for k, v in locals().items() if v is variable][0]

    def bag_of_words(self):
        bag_words = set(word for row in self.data_set['Feature'] for word in row)
        print('Created bag of words. Length:{0}'.format(len(bag_words)))
        fs_words = pd.DataFrame([([(word in data['Feature']) for word in bag_words], data['Label']) for _, data in self.data_set.iterrows()], columns=['Feature', 'Label'])
        print('Created Featureset "{0}" for {1} documents.'.format('words', len(fs_words)))
        return bag_words, fs_words

    def bag_of_bigrams(self):
        bag_bigrams = set(gram for row in self.data_set['Feature'] for gram in ngrams(row, 2))
        print('Created bag of bigrams. Length:{0}'.format(len(bag_bigrams)))
        fs_bigrams = pd.DataFrame([([gram in ngrams(data['Feature'], 2) for gram in bag_bigrams], data['Label']) for _, data in self.data_set.iterrows()], columns=['Feature', 'Label'])
        print('Created Featureset "{0}" for {1} documents.'.format('bigrams', len(fs_bigrams)))
        return bag_bigrams, fs_bigrams

    def bag_of_words_min(self, min_count=5):
        assert self.bag_words is not None, "Please build bag of words first!"

        min_word_count = min_count

        sents = self.data_set['Feature'].values.tolist()
        sent_merged = [j for i in sents for j in i]
        word_count = Counter(sent_merged)

        bag_words_min = set(word for word in self.bag_words if word_count[word] >= min_word_count)
        print('Created bag of words with minimum count {0}. Length: {1}'.format(min_word_count, len(bag_words_min)))

        min_word_frame = deepcopy(self.data_set)
        min_word_frame['Feature_min'] = ''

        for index, row in min_word_frame.iterrows():
            f = []
            for word in row['Feature']:
                if word_count[word] >= min_word_count:
                    f.append(word)
            min_word_frame.at[index, 'Feature_min'] = f

        fs_words_min = pd.DataFrame([([(word in data['Feature_min']) for word in bag_words_min], data['Label']) for _, data in min_word_frame.iterrows()], columns=['Feature', 'Label'])
        print('Created Featureset "{0}" for {1} documents.'.format('words min', len(fs_words_min)))

        return bag_words_min, fs_words_min

    def pos_tags(self):
        bag_tags = set(tag for row in self.data_set['PosTags'] for tag in row)
        print('Created bag of POS-tags. Length:{0}'.format(len(bag_tags)))
        fs_pos = pd.DataFrame([([(tag in row['PosTags']) for tag in bag_tags], row['Label']) for _, row in self.data_set.iterrows()], columns=['Feature', 'Label'])
        print('Created Featureset "{0}" for {1} documents.'.format('pos', len(fs_pos)))

        return bag_tags, fs_pos


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

        bloblist = [tb(i) for i in data_set_sent]
        tfidf_list = []
        for i, blob in enumerate(bloblist):
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            for word in scores:
                tfidf_frame.at[i, word] = round(scores[word], 5)
            tfidf_list.append(scores)

        tfidf_values = tfidf_frame.values.tolist()
        fs_tfidf_words = deepcopy(self.fs_words)
        fs_tfidf_words['Feature'] = tfidf_values
        self.data_set['TfIdf'] = tfidf_list
        print('Created Featureset "{0}" for {1} documents.'.format('tfidf', len(fs_tfidf_words)))

        return fs_tfidf_words

    def word2vec(self):
        assert self.bag_words is not None and self.fs_words is not None, "Please build bag of words and fs words first!"
        assert 'TfIdf' in self.data_set.columns, 'No TfIdf Vectors found. Please calculate Tfidf Vectors first!'

        import gensim
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format('..\\german.model', binary=True, )
        print('Loaded word2vec model.')

        data_set_w2v = deepcopy(self.data_set)
        data_set_w2v['Word2Vec'] = ''

        def get_w2v(word):
            try:
                vec = w2v_model.get_vector(word)
                return np.round(vec, 8)
            except:
                vec = np.zeros(shape=(1, 300))
                return vec[0]

        bag_w2v = {}
        for word in self.bag_words:
            bag_w2v[word] = get_w2v(word)
        print('Created bag of Word2Vec Vectors. Length:{0}'.format(len(bag_w2v)))


        for i, row in data_set_w2v.iterrows():
            avg_vecs = []
            word_len = 0
            for j, word in enumerate(row['Feature'][:-1]):
                vec = bag_w2v[word]
                if not sum(vec) == 0:
                    word_len += 1

                avg_vecs.append(list(vec * row['TfIdf'][word]))

            if word_len > 0:
                data_set_w2v.at[i, 'Word2Vec'] = np.array(avg_vecs).sum(axis=0) / word_len
            else:
                data_set_w2v.at[i, 'Word2Vec'] = np.array(avg_vecs).sum(axis=0)

        w2v_values = list(np.round(data_set_w2v['Word2Vec'].values.tolist(), 6))
        fs_w2v = deepcopy(self.fs_words)
        fs_w2v['Feature'] = w2v_values
        print('Created Featureset "{0}" for {1} documents.'.format('word2vec', len(fs_w2v)))

        return fs_w2v

    def doc2vec(self, return_model=False):
        import gensim
        def read_corpus(data, tokens_only=False):
            for i, line in data.iterrows():
                if tokens_only:
                    yield gensim.models.doc2vec.TaggedDocument(line['Feature'])
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(line['Feature'], tags=[line['Label']])

        shuffled = self.data_set.sample(frac=1)
        corpus = list(read_corpus(shuffled))

        model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, epochs=40)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        print('Successfully trained Doc2Vec model on {0} documents.'.format(len(corpus)))
        fs_d2v = deepcopy(self.data_set)
        for i, row in fs_d2v.iterrows():
            fs_d2v.at[i, 'Feature'] = model.infer_vector(row['Feature'])

        print('Created Featureset "{0}" for {1} documents.'.format('doc2vec', len(fs_d2v)))

        if return_model:
            return fs_d2v, model
        else:
            return fs_d2v


    def combine(self, sets):
        new = deepcopy(sets[0])
        for i in sets[:1]:
            new['Feature'] = new['Feature'] + i['Feature']
        return new


    def export(self, size):
        pickles.create_pickle(self, '\\feature_sets\\fs_' + str(size))

    def load_from_file(self, size):
        self = pickles.load_pickle('\\feature_sets\\fs_' + str(size))
