import pandas as pd
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
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

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

def data_processing(data, labels, validation_split):
    vocab_size = 20000

    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Get max training sequence length
    max_sequence_length = max([len(x) for x in sequences])
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    print(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = validation_split, random_state=1)
    print(X_train)

    return X_train, X_test, y_train, y_test, max_sequence_length, word_index, vocab_size

text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, word_index, vocab_size = data_processing(text_df['review'], text_df['sentiment'],0.3)

###################
# BUILD THE MODEL #
###################

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(vocab_size, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

print(len(X_train), "Training sequences")
print(len(X_test), "Validation sequences")

################################
# TRAIN AND EVALUATE THE MODEL #
################################
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test, y_test))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()