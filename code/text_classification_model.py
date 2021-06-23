from nltk import text
import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from bs4 import BeautifulSoup
import re

from textblob import TextBlob
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix


text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")

print(text_df.head())
#################
# PREPROCESSING #
#################
def data_preprocessing(dataframe, target_column):
    # Dropping empty cells from dataframe
    dataframe = dataframe.dropna()

    # Coverting text to lowercase
    target_column = target_column.str.lower()

    # Converting emojis to words
    for emot in UNICODE_EMO:
        target_column = target_column.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))

    #for emot in EMOTICONS:
    #    target_column = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), target_column)
    
    # Defining regular expressions:
    url_regex = re.compile(r'https?://\S+|www\.\S+') # Website regular expression, eg: www.example.com
    mentions_regex = re.compile(r'@[A-Za-z0-9]+') # Mentions regular expression, eg: @example
    emails_regex = re.compile(r'[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+') # Emails regular expression, eg: example@example.com
    punctuation_regex = re.compile(r'[^\w\s]') # Punctuation regular expression, eg: ; ? ' etc...

    # Applying the above defined regular expressions to the target column:
    target_column = target_column.str.replace(url_regex,'',regex=True)
    target_column = target_column.str.replace(mentions_regex,'',regex=True)
    target_column = target_column.str.replace(emails_regex,'',regex=True)
    target_column = target_column.str.replace(punctuation_regex,'',regex=True)

    # Removing html
    for text in target_column:
        BeautifulSoup(text,'lxml').target_column

    # Importing english stopwords from nltk library 
    eng_stopwords = set(stopwords.words('english'))
    # Function to remove stopwords
    target_column = target_column.apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))

    return dataframe, target_column

text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])

print(text_df.head())