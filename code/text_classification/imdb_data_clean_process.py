import re
from emot.emo_unicode import EMOTICONS, UNICODE_EMO
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np

#################
# DATA CLEANING #
#################
def data_preprocessing(dataframe, x_column):
    print("Processing data...")

    # Dropping empty cells from dataframe
    dataframe = dataframe.dropna()

    # Coverting text to lowercase
    x_column = x_column.str.lower()

    # Converting Emojis and emoticons into words
    for emot in UNICODE_EMO:
        x_column = x_column.replace(emot, "_".join(UNICODE_EMO[emot].replace(","," ").replace(":"," ").split()))
    for emot in EMOTICONS:
        x_column = x_column.replace(u'('+emot+')', "_".join(EMOTICONS[emot].replace(","," ").split()))
    
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
    x_column = x_column.apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))

    """# Lemmatization, reducing inflected words to their word stem ensuring the root word belongs to the language
    lemmatizer = WordNetLemmatizer()

    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb
    # Function for lemmatization using POS tag
    def lemmatize_text(text):
        pos_tagged_text = pos_tag(text.split())
        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
        
    x_column = x_column.apply(lemmatize_text)"""

    return dataframe, x_column

###################
# DATA PROCESSING #
###################
def data_processing(data, labels, validation_split):
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
