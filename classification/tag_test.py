import pickle
import nltk
from nltk.util import ngrams
from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

with open('nltk_german_classifier_data.pickle', 'rb') as f:
    tagger = pickle.load(f)

def bag_of_words(words):
    return dict([(word, True) for word in words])

import csv
reader = csv.reader(open('classified.csv', 'r'), delimiter=';')
for line in reader:
    sent = nltk.tokenize.WordPunctTokenizer().tokenize(line[0])
    print(tagger.tag(sent))
    print(bag_of_words((sent)))
    ngram = ngrams(sent, 2)
    for gram in ngram:
        print(gram)

pass
