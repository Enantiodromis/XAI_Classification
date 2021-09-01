# Author: George Richard Edward Bradley
# Project: xai_classification_mixed_data
# Title: text_dataset_3_processing
# GitHub: https://github.com/Enantiodromis

###########
# IMPORTS #
###########
import re

import pandas as pd
from emot.emo_unicode import EMOTICONS, UNICODE_EMO
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.ops.gen_array_ops import inplace_add

#################
# DATA BUILD #
#################
def dataset_3_build():
    print("Fetching data...")

    # Reading in both data as dataframes
    df_data_3 = pd.read_csv("datasets/text_data/text_data_3/amazon_yelp_twitter.csv")
    df_data_3.columns = ['sentiment', 'text']
    
    # Returning dataframe object
    return df_data_3

#################
# DATA CLEANING #
#################
def dataset_3_preprocessing(dataframe):
    print("Processing data...")

    # Dropping empty cells from dataframe
    dataframe.dropna(how='any', inplace=True)

    # Isolating the column with "X_data"
    x_column = dataframe['text']
    # Isolating the column with "Y_data"
    y_data = dataframe['sentiment']

    # Coverting text to lowercase
    x_column = x_column.str.lower()
    x_column.dropna(inplace = True)
    
    # Defining regular expressions:
    html_regex = re.compile(r'<.*?>') # HTML tag regular expression, <br>  
    url_regex = re.compile(r'https?://\S+|www\.\S+') # Website regular expression, eg: www.example.com
    mentions_regex = re.compile(r'@[A-Za-z0-9]+') # Mentions regular expression, eg: @example
    emails_regex = re.compile(r'[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+') # Emails regular expression, eg: example@example.com
    punctuation_regex = re.compile(r'[^\w\s]') # Punctuation regular expression, eg: ?, ' , ; etc...
    numbers_regex = re.compile(r'\b\d+(?:\.\d+)?\s+') # Number regular expression, eg: 3

    # Applying the above defined regular expressions to the target column:
    x_column = x_column.str.replace(html_regex,'',regex=True)
    x_column = x_column.str.replace(url_regex,'',regex=True)
    x_column = x_column.str.replace(mentions_regex,'',regex=True)
    x_column = x_column.str.replace(emails_regex,'',regex=True)
    x_column = x_column.str.replace(punctuation_regex,'',regex=True)
    x_column = x_column.str.replace(numbers_regex,'',regex=True)

    # Importing english stopwords from nltk library and removing from dataframe
    eng_stopwords = set(stopwords.words('english'))
    x_column = x_column.apply(lambda x: [word for word in x.split() if word not in (eng_stopwords)])

    # Returning rows associated with X and Y data
    return x_column, y_data

###################
# DATA PROCESSING #
###################
def get_dataset_3(validation_split, sample_size):
    
    # Calling the build function and using the returned value to call the preprocessing function
    df_3 = dataset_3_build()
    sampled_df_3 = df_3.sample(sample_size)
    X_data, y_data = dataset_3_preprocessing(sampled_df_3)

    print(X_data.shape[0])
    print(y_data.shape[0])

    print("Tokenising and splitting data...")

     # The returned X and Y data from the preprocessing function is used in the get_dataset function
    
    # Creating train and test data using train_test_split, the validatation split is defined by the user.
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = validation_split, random_state=1)

    # Defining a vocab_size of 20000
    vocab_size = 20000

    # Initialising the tokenizer with the defined vocab_size
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    # Fitting the tokenizer to the X data
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # Defining the word index
    word_index = tokenizer.word_index

    # Defining a max training sequence length
    max_sequence_length = max([len(x) for x in sequences_train])
    X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
    X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

    # Initialising a binarizer
    encoder = LabelBinarizer()
    # Fitting the binarizer to the labels of both the train and test data
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    
    return X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer
