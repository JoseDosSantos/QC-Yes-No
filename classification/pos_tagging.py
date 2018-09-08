import nltk
import random

corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09', ['ignore', 'words', 'ignore', 'ignore', 'pos'], encoding='utf-8')

tagged_sents = list(corp.tagged_sents())
random.shuffle(tagged_sents)

split_perc = 0.1
split_size = int(len(tagged_sents) * split_perc)
train_sents, test_sents = tagged_sents, tagged_sents[:split_size]


from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
tagger = ClassifierBasedGermanTagger(train=train_sents)

accuracy = tagger.evaluate(test_sents)
print(accuracy)

import pickle

with open('nltk_german_classifier_data.pickle', 'wb') as f:
    pickle.dump(tagger, f, protocol=2)