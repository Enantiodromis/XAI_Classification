import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import re

from textblob import TextBlob
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")

#################
# PREPROCESSING #
#################
def data_preprocessing(dataframe, target_column):
    # Dropping empty cells from dataframe
    dataframe = dataframe.dropna()

    # Coverting text to lowercase
    target_column = target_column.str.lower()

    # Converting Emojis and emoticons into words
    for emot in UNICODE_EMO:
        target_column = target_column.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
    for emot in EMOTICONS:
        target_column = target_column.replace(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()))
    
    # Defining regular expressions:
    html_regex = re.compile(r'<.*?>') # HTML tag regular expression, <br>  
    url_regex = re.compile(r'https?://\S+|www\.\S+') # Website regular expression, eg: www.example.com
    mentions_regex = re.compile(r'@[A-Za-z0-9]+') # Mentions regular expression, eg: @example
    emails_regex = re.compile(r'[A-Za-z0-9]+@[a-zA-z].[a-zA-Z]+') # Emails regular expression, eg: example@example.com
    punctuation_regex = re.compile(r'[^\w\s]') # Punctuation regular expression, eg: ; ? ' etc...

    # Applying the above defined regular expressions to the target column:
    target_column = target_column.str.replace(html_regex,'',regex=True)
    target_column = target_column.str.replace(url_regex,'',regex=True)
    target_column = target_column.str.replace(mentions_regex,'',regex=True)
    target_column = target_column.str.replace(emails_regex,'',regex=True)
    target_column = target_column.str.replace(punctuation_regex,'',regex=True)

    # Importing english stopwords from nltk library and removing from dataframe
    eng_stopwords = set(stopwords.words('english'))
    target_column = target_column.apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))

    # Lemmatization, reducing inflected words to their word stem ensuring the root word belongs to the language
    lemmatizer = WordNetLemmatizer()

    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb
    # Function for lemmatization using POS tag
    def lemmatize_words(text):
        pos_tagged_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
        
    target_column = target_column.apply(lemmatize_words)

    return dataframe, target_column

text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])

print(text_df.head())