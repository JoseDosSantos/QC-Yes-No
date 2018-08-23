import nltk
import csv
import random

FILE_ONE = 'tweets/output_1000_tweets_2018-08-22_16-25-32/classified.csv'
FILE_TWO = 'tweets/output_1000_tweets_2018-08-21_17-02-43/classified.csv'
FILE_THREE = 'tweets/output_1000_tweets_2018-08-21_16-25-20/classified.csv'
FILE_FOUR = 'tweets/output_1000_tweets_2018-08-21_21-35-57/classified.csv'
FILE_FIVE = 'tweets/output_1000_tweets_2018-08-23_09-37-19/classified.csv'

FULL_SET = [FILE_ONE, FILE_TWO, FILE_THREE, FILE_FOUR, FILE_FIVE]

# Read data from all input files
data_set = []
for file in FULL_SET:
    reader = csv.reader(open(file, 'r'), delimiter=';')
    for line in reader:
        data_set.append(line)

# Create the set of all words and bag-of-words
all_words_train = set(word.lower() for passage in data_set for word in nltk.tokenize.WordPunctTokenizer().tokenize(passage[0]))
feature_set = [({word: (word in nltk.tokenize.WordPunctTokenizer().tokenize(data[0])) for word in all_words_train}, data[1]) for data in data_set]
random.shuffle(feature_set)

# Create Training and Test sets
total = len(feature_set)
training_percentage = 0.9
train_set, test_set = feature_set[:int(total * training_percentage)], feature_set[int(total * training_percentage):]


# Train the classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features()
print(nltk.classify.accuracy(classifier, train_set))



# Create 