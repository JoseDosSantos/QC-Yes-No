import tweepy
import re
import nltk
import os
import csv
from _datetime import datetime


print(os.path.dirname(os.path.realpath(__file__)))
nltk.data.path.append('C:\\Users\\jcersovsky\\Documents\\Projects\\Bachelorarbeit\\nltk_data')

consumer_key = "363klSkj4Eb6cy36ZFwc1zLwN"
consumer_secret = "xSnNdpeicNyGmK7mredAK0pCKhoTUOX135bcv1ksxgKAtg5n7h"
access_token = "1594193462-keWE3IhUAejzwlJSQgJv8oEmjAzmVXEgk0lP9Xa"
access_token_secret = "NE6pm63qZDkYI6xfTpXR9kITWXfyefgNNvUHlmk5f3Dyy"

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

if (not api):
    print("Can't Authenticate")
    exit(-1)


# The search term you want to find
query = "?"
# Language code (follows ISO 639-1 standards)
language = "de"
# Total number of tweets
maxTweets = 1000
# This is the max the API permits
tweetsPerQry = 100
# Type of the output file
file_type = '.csv'
# Output folder
folder = '..\\tweets\\'


# Calling the user_timeline function with our parameters
#results = api.search(q=query, lang=language, count=maxTweets, show_user=False, tweet_mode='extended')

def get_file_name():
    return(folder + str(maxTweets) + '_tweets_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + file_type)




def remove_link(tweet):
    return re.sub(r'https:\/\/t\.co\/[\w\d]*', '', tweet)

def remove_rt(tweet):
    return re.sub(r'RT\s@\w+:\s', '', tweet)

def remove_at(tweet):
    return re.sub(r'@[\w\d]+', '', tweet)

def remove_semicolon(tweet):
    return re.sub(r';', '', tweet)

def tokenize_sentences(tweet):
    tokenizer = nltk.data.load('file://C:/Users/Josef/PycharmProjects/QC-Yes-No/nltk_data/tokenizers/punkt/german.pickle')
    return tokenizer.tokenize(tweet)



def clean_tweet(original_tweet):
    tweet = remove_link(original_tweet)
    tweet = remove_rt(tweet)
    tweet = remove_at(tweet)
    tweet = remove_semicolon(tweet)
    tweet = tokenize_sentences(tweet)
    return tweet



def get_tweets():
    tweetCount = 0
    sinceId = None
    max_id = -1
    valid_questions = 0
    with open(get_file_name(), 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter=';')
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=query, count=tweetsPerQry, lang=language)
                    else:
                        new_tweets = api.search(q=query, count=tweetsPerQry, lang=language, since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=query, count=tweetsPerQry, lang=language, max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=query, count=tweetsPerQry, lang=language, max_id=str(max_id - 1), since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    for sent in clean_tweet(tweet._json['text']):
                        if '?' in sent:
                            valid_questions += 1
                            if len(sent) > 10:
                                try:
                                    writer.writerow([sent.strip()])
                                except UnicodeEncodeError:
                                    print('Skipped tweet due to unicode error on tweet {}'.format(valid_questions))


                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break
        print('Valid questions: ' + str(valid_questions))


