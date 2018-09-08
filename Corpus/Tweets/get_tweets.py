import tweet_extraction
import time
from datetime import datetime, timedelta

def get_time():
    return datetime.now()
try:
    while True:
        print('Starting collection at: ' + get_time().time().strftime('%H:%M:%S'))
        tweet_extraction.get_tweets()
        print('Finished extraction. Next request in 15 minutes at {0}'.format((get_time() + timedelta(minutes=15)).time().strftime('%H:%M:%S')))
        time.sleep(900)
        continue
except KeyboardInterrupt:
    print('Collection stopped.')