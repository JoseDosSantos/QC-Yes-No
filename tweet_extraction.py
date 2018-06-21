import tweepy
import re
import nltk

nltk.data.path.append('C:\\Users\\jcersovsky\\PycharmProjects\\QC\\nltk_data')

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

# The search term you want to find
query = "?"
# Language code (follows ISO 639-1 standards)
language = "de"

# Calling the user_timeline function with our parameters
results = api.search(q=query, lang=language, count=200, show_user=False, tweet_mode='extended')



def remove_link(tweet):
    return re.sub(r'https:\/\/t\.co\/[\w\d]*', '', tweet)

def remove_rt(tweet):
    return re.sub(r'RT\s@\w+:\s', '', tweet)

def tokenize_sentences(tweet):
    return nltk.sent_tokenize(tweet, 'german')

def clean_tweet(original_tweet):
    tweet = remove_link(original_tweet)
    tweet = remove_rt(tweet)
    tweet = tokenize_sentences(tweet)
    return tweet



# foreach through all tweets pulled
with open("Output.txt", "w", encoding="utf-8") as text_file:
    for tweet in results:
        cleaned_tweet = clean_tweet(tweet._json['full_text'])
        # printing the text stored inside the tweet object
        for sent in cleaned_tweet:
            if '?' in sent:
                print(sent + '\n', file=text_file)
        print('\n' + 20*'*' + '\n', file=text_file)






