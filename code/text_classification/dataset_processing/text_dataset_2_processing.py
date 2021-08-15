import numpy as np
import pandas as pd
import re
from emot.emo_unicode import EMOTICONS, UNICODE_EMO
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def dataset_2_build():
    print("Building data...")
    fake_news = pd.read_csv("datasets/text_data/text_data_2/Fake.csv")
    fake_news["fake"] = 1
    real_news = pd.read_csv("datasets/text_data/text_data_2/True.csv")
    real_news["fake"] = 0

    news = pd.concat([fake_news, real_news])

    news['text'] = news['title'] + news['text']
    news.drop(labels=['title'], axis=1, inplace=True)

    news.drop(labels=['subject', 'date'], axis=1, inplace=True)

    news = news.sample(frac=1)
    feature_text = news['text']
    target = news['fake']

    return news

def dataset_2_data_preprocessing(dataframe):
    print("Processing data...")
    
    news = dataframe.sample(frac=1)
    feature_text = news['text']
    target = news['fake']

    # Dropping empty cells from dataframe
    dataframe = dataframe.dropna()

    # Coverting text to lowercase
    x_column = feature_text.str.lower()
    x_column.dropna(inplace=True)
    
    # Defining regular expressions:
    html_regex = re.compile(r'<.*?>') # HTML tag regular expression, <br>  
    url_regex = re.compile(r'https?://\S+|www\.\S+') # Website regular expression, eg: www.example.com
    mentions_regex = re.compile(r'@[A-Za-z0-9]+') # Mentions regular expression, eg: @example
    emails_regex = re.compile(r'[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+') # Emails regular expression, eg: example@example.com
    punctuation_regex = re.compile(r'[^\w\s]') # Punctuation regular expression, eg: ?, ' , ; etc...
    numbers_regex = re.compile(r'\b\d+(?:\.\d+)?\s+') # Number regular expression, eg: 3

    # Applying the above defined regular expressions to the target column:
    x_column = x_column.str.replace(html_regex,' ',regex=True)
    x_column = x_column.str.replace(url_regex,' ',regex=True)
    x_column = x_column.str.replace(mentions_regex,' ',regex=True)
    x_column = x_column.str.replace(emails_regex,' ',regex=True)
    x_column = x_column.str.replace(punctuation_regex,' ',regex=True)
    x_column = x_column.str.replace(numbers_regex,' ',regex=True)

    # Importing english stopwords from nltk library and removing from dataframe
    eng_stopwords = set(stopwords.words('english'))
    x_column = x_column.apply(lambda x: [word for word in x.split() if word not in (eng_stopwords)])

    return x_column, target

def dataset_2_data_processing(data, labels, validation_split):
    print("Loading and splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = validation_split, random_state=1)
    
    X_train_cpy = X_train
    X_test_cpy = X_test
    y_test_cpy = y_test

    vocab_size = 20000
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # Defining the word index
    word_index = tokenizer.word_index
    print("Unique words: {}".format(len(word_index)))

    # Get max training sequence length
    #max_sequence_length = max([len(x) for x in sequences_train])
    max_sequence_length = 30
    X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
    X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    
    print('Shape of x train tensor:', X_train.shape)
    print('Shape of x test tensor:', X_test.shape)
    print('Shape of y train tensor:', y_train.shape)
    print('Shape of y test tensor:', y_test.shape)
    print('Classes:',encoder.classes_)
    
    return X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer