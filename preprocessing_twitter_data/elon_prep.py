import pandas as pd
import re
from sklearn.model_selection import train_test_split

def cleaning_pic_url (text):
    text = re.sub(r'pic.twitter.com/[\w]*',"", text)
    return text
def cleaning_quotes(text):
    text = re.sub(r'--do[\w]*',"", text)
    return text

def cleaning_mentions(text):
    text = re.sub("@[A-Za-z0-9_]+","", text)
    return text


def get_data(file):
    df = pd.read_csv(file)
    ## Making it all lower case
    df.tweet = df.tweet.str.lower()
    df['contents_w/0_http'] = df['tweet'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])
    df['contents_w/0_https'] = df['contents_w/0_http'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    df['contents_w/0_https'] =  df['contents_w/0_https'].apply(lambda x: cleaning_pic_url(x))
    df['contents_w/0_https'] = df['contents_w/0_https'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    tweets = df['contents_w/0_https']


    X_train, X_rem = train_test_split(tweets,train_size=0.8)
    X_valid, X_test= train_test_split(X_rem,test_size=0.5)
    
    X_train.to_csv(r'twitter_data/train/elon_musk.txt', header=None, index=None, sep='\t', mode='a')
    X_valid.to_csv(r'twitter_data/valid/elon_musk.txt', header=None, index=None, sep='\t', mode='a')
    X_test.to_csv(r'twitter_data/test/elon_musk.txt', header=None, index=None, sep='\t', mode='a')
    # tweets.to_csv(r'raw_data/preprocessed_elon.csv', header=None, index=None, sep='\t', mode='a')

    # tweets.to_csv(r'twitter_data/preprocessed_donaldtrump.txt', header=None, index=None, sep='\t', mode='a')




get_data('raw_data/elon.csv')