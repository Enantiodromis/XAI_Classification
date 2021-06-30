import pandas as pd

import numpy as np
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout

from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")
print(text_df.head())

#################
# DATA CLEANING #
#################

def data_preprocessing(dataframe, x_column):
    # Dropping empty cells from dataframe
    dataframe = dataframe.dropna()

    # Coverting text to lowercase
    x_column = x_column.str.lower()

    # Converting Emojis and emoticons into words
    for emot in UNICODE_EMO:
        x_column = x_column.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    for emot in EMOTICONS:
        x_column = x_column.replace(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()))
    
    # Defining regular expressions:
    html_regex = re.compile(r'<.*?>') # HTML tag regular expression, <br>  
    url_regex = re.compile(r'https?://\S+|www\.\S+') # Website regular expression, eg: www.example.com
    mentions_regex = re.compile(r'@[A-Za-z0-9]+') # Mentions regular expression, eg: @example
    emails_regex = re.compile(r'[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+') # Emails regular expression, eg: example@example.com
    punctuation_regex = re.compile(r'[^\w\s]') # Punctuation regular expression, eg: ; ? ' etc...

    # Applying the above defined regular expressions to the target column:
    x_column = x_column.str.replace(html_regex,'',regex=True)
    x_column = x_column.str.replace(url_regex,'',regex=True)
    x_column = x_column.str.replace(mentions_regex,'',regex=True)
    x_column = x_column.str.replace(emails_regex,'',regex=True)
    x_column = x_column.str.replace(punctuation_regex,'',regex=True)

    # Importing english stopwords from nltk library and removing from dataframe
    eng_stopwords = set(stopwords.words('english'))
    x_column = x_column.apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))

    # Lemmatization, reducing inflected words to their word stem ensuring the root word belongs to the language
    lemmatizer = WordNetLemmatizer()

    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb
    # Function for lemmatization using POS tag
    def lemmatize_text(text):
        pos_tagged_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
        
    x_column = x_column.apply(lemmatize_text)

    return dataframe, x_column

###################
# DATA PROCESSING #
###################
def data_processing(x_data, y_data, test_proportion):
    # Splitting data (train and test)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_proportion, random_state = 0)

    # The data set has tens or hundreds of thousands of unique words, we need to limit this number.
    # We initialise a tokenizer and use vocab_size to keep the most frequent 20,000 words.
    vocab_size = 20000
    num_classes = 2

    tokenizer = Tokenizer(num_words = vocab_size)
    x_train = tokenizer.texts_to_matrix(x_train)
    x_test = tokenizer.texts_to_matrix(x_test)

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    return x_train, x_test, y_train, y_test, vocab_size, num_classes

text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])
print(text_df.head())

x_train, x_test, y_train, y_test, vocab_size, num_classes, = data_processing(text_df['review'], text_df['sentiment'],0.3)








