import tweepy
import re
import nltk

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
# Name of the output file
fName = 'tweets.txt'


# Calling the user_timeline function with our parameters
#results = api.search(q=query, lang=language, count=maxTweets, show_user=False, tweet_mode='extended')



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

tweetCount = 0
sinceId = None
max_id = -1

with open(fName, 'w', encoding="utf-8") as f:
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
                        f.write(sent + '\n')

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break
pass



# # foreach through all tweets pulled
# with open(fName, "w", encoding="utf-8") as text_file:
#     for tweet in results:
#         cleaned_tweet = clean_tweet(tweet._json['full_text'])
#         # printing the text stored inside the tweet object
#         for sent in cleaned_tweet:
#             if '?' in sent:
#                 print(sent + '\n', file=text_file)
#         print('\n' + 20*'*' + '\n', file=text_file)






