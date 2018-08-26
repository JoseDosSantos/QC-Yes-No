import nltk
import csv
import random
from nltk.util import ngrams


FILE_ONE = 'tweets/output_1000_tweets_2018-08-22_16-25-32/classified.csv'
FILE_TWO = 'tweets/output_1000_tweets_2018-08-21_17-02-43/classified.csv'
FILE_THREE = 'tweets/output_1000_tweets_2018-08-21_16-25-20/classified.csv'
FILE_FOUR = 'tweets/output_1000_tweets_2018-08-21_21-35-57/classified.csv'
FILE_FIVE = 'tweets/output_1000_tweets_2018-08-23_09-37-19/classified.csv'

FULL_SET = [FILE_ONE, FILE_TWO]#, FILE_THREE, FILE_FOUR, FILE_FIVE]

# Read data from all input files
data_set = []
for file in FULL_SET:
    reader = csv.reader(open(file, 'r'), delimiter=';')
    for line in reader:
        data_set.append([line[0].lower(), line[1]])

# # Create the set of all words and bag-of-words
# bow_words_bag = set(word for passage in data_set for word in nltk.tokenize.WordPunctTokenizer().tokenize(passage[0]))
# bow_feature_set = [({word: (word in nltk.tokenize.WordPunctTokenizer().tokenize(data[0])) for word in bow_words_bag}, data[1]) for data in data_set]
# random.shuffle(bow_feature_set)
#
# # Create Training and Test sets
total = len(data_set)
training_percentage = 0.9
# bow_train_set, bow_test_set = bow_feature_set[:int(total * training_percentage)], bow_feature_set[int(total * training_percentage):]
#
#
# # Train the classifier
# bow_classifier = nltk.NaiveBayesClassifier.train(bow_train_set)
# bow_classifier.show_most_informative_features()
# print(nltk.classify.accuracy(bow_classifier, bow_test_set))


# Create bag of n-grams (bigrams)
bon_words_bag = set(gram for passage in data_set for gram in ngrams(nltk.tokenize.WordPunctTokenizer().tokenize(passage[0]), 2))
bon_feature_set = [({gram: (gram in ngrams(nltk.tokenize.WordPunctTokenizer().tokenize(data[0]), 2)) for gram in bon_words_bag}, data[1]) for data in data_set]
random.shuffle(bon_feature_set)
bon_train_set, bon_test_set = bon_feature_set[:int(total * training_percentage)], bon_feature_set[int(total * training_percentage):]
bon_classifier = nltk.NaiveBayesClassifier.train(bon_train_set)
bon_classifier.show_most_informative_features()
print(nltk.classify.accuracy(bon_classifier, bon_test_set))


