from keras.engine.sequential import Sequential
import pandas as pd
import numpy as np
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as backend
#tf.compat.v1.disable_v2_behavior()

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")

print(tf.__version__)
print(tf.config.list_physical_devices())

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
    max_sequence_length = max([len(x) for x in sequences_train])
    X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
    X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    print('Shape of x train tensor:', X_train.shape)
    print('Shape of x test tensor:', X_test.shape)
    print('Shape of y train tensor:', y_train.shape)
    print('Shape of y test tensor:', y_test.shape)
    print(y_test)

    return X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy

#######################################
# BUILD, TRAIN AND EVALUATE THE MODEL #
#######################################

def lstm_model(vocab_size, X_train, y_train, X_test, y_test, number_epochs):
    # Input shape is defined as shape(None,) for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Each integer is embeded as a 128-dimensional vector
    x = layers.Embedding(vocab_size, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    # Add a classifier
    outputs = layers.Dense(1, activation="sigmoid")(x) # A sigmoid activation function as we are classifying binary data
    model = keras.Model(inputs, outputs)
    model.summary() # Printing 
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, batch_size=32, epochs=number_epochs, validation_data=(X_test, y_test))

    # Saving the model
    model.save("models/LSTM_%d" %epoch_num, save_format="h5")
    
    return history, model

####################################
# PLOTTING MODEL'S ACCURACY & LOSS #
####################################

def plot_accuracy_loss(model_history, number_epochs):
    x_list = []
    x_list.extend(range(0,number_epochs))

    plt.figure(1)
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('data_plots/model_accuracy.jpg')

    plt.figure(2)
    plt.plot(model_history.history['loss'], color='green')
    plt.plot(model_history.history['val_loss'], color='red')
    plt.title('Model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('data_plots/model_loss.jpg')

##################
# SHAP EXPLAINER #
##################

def shap_explainer(X_train_not_encoded, X_test_not_encoded, X_train_encoded, X_test_encoded, model, word_index, y_test_cpy):
    y_test_cpy = np.array(y_test_cpy).flatten()
    background = X_train_encoded[:100]
    session = backend.get_session()
    explainer = shap.DeepExplainer(model, background, session)

    num_explanation = 10
    print(X_test_not_encoded[:num_explanation])  
    shap_values = explainer.shap_values(X_test_encoded[:num_explanation])
    print(shap_values)
    """num2word = {}
    for w in word_index.keys():
        num2word[word_index[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), X_test_encoded[i]))) for i in range(10)])
    print("X_test_words: ", x_test_words)
    shap.summary_plot(shap_values, feature_names = list(num2word.values()))
    #shap.summary_plot(shap_values, feature_names = list(num2word.values()))"""

    """# init the JS visualization code
    shap.initjs()
    # create dict to invert word_idx k,v order
    num2word = {}
    for w in word_index.keys():
        num2word[word_index[w]] = w
    x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), X_test_encoded[i]))) for i in range(10)])

    # plot the explanation of a given prediction
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test_words[0])"""

epoch_num = 11
#model = load_model("models/LSTM_%d" %epoch_num)

text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy = data_processing(text_df['review'], text_df['sentiment'],0.3)
history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, epoch_num)
plot_accuracy_loss(history, epoch_num)
#shap_explainer(X_train_cpy, X_test_cpy, X_train, X_test, model, word_index, y_test_cpy)



