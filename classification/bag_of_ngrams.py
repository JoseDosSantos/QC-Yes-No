import nltk
import csv
import random
from nltk.util import ngrams
import config
from sklearn.svm import LinearSVC


# Read data from all input files
data_set = []
for file in config.FULL_SET:
    reader = csv.reader(open(file, 'r'), delimiter=';')
    for line in reader:
        data_set.append([line[0].lower(), line[1]])

total = len(data_set)
training_percentage = 0.9

# Create feature sets and divide into training and test
words_bag = set(gram for passage in data_set for gram in ngrams(nltk.tokenize.WordPunctTokenizer().tokenize(passage[0]), 2))
feature_set = [({gram: (gram in ngrams(nltk.tokenize.WordPunctTokenizer().tokenize(data[0]), 2)) for gram in words_bag}, data[1]) for data in data_set]
random.shuffle(feature_set)
train_set, test_set = feature_set[:int(total * training_percentage)], feature_set[int(total * training_percentage):]

# Naive Bayes Classifier
naive_bayes_enabled = False
if naive_bayes_enabled:
    bon_classifier = nltk.NaiveBayesClassifier.train(train_set)
    bon_classifier.show_most_informative_features()
    print(nltk.classify.accuracy(bon_classifier, test_set))

# Support Vector Machine
support_vector_enabled = False
if support_vector_enabled:
    classifier = nltk.classify.SklearnClassifier(LinearSVC())
    classifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))

# Decision Tree
decision_tree_enabled = True
if decision_tree_enabled:
    classifier = nltk.DecisionTreeClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
