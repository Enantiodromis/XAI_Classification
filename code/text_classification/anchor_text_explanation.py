import os 
import os.path
from matplotlib.pyplot import text
import numpy as np 
from dataset_processing.text_dataset_1_processing import imdb_data_preprocessing, imdb_data_processing
from dataset_processing.text_dataset_2_processing import dataset_2_build, dataset_2_data_preprocessing, dataset_2_data_processing
from dataset_processing.text_dataset_3_processing import dataset_3_data_preprocessing, dataset_3_data_processing
from text_classification_core import lstm_model, plot_accuracy_loss
import spacy 
import sys
from anchor.anchor_text import AnchorText
import keras.backend as backend
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd

####################
# ANCHOR EXPLAINER #
####################
def anchor_explainer(X_test_encoded, model, word_index, tokenizer):
    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test_encoded)))

    def wrapped_predict(strings):
        cnn_rep = tokenizer.texts_to_sequences(strings)
        text_data = pad_sequences(cnn_rep, maxlen=30)
        prediction = model.predict(text_data)
        predicted_class = np.where(prediction > 0.5, 1,0)[0]
        return predicted_class

    test_text = ' '.join(my_texts[6])
    nlp = spacy.load('en_core_web_sm')
    explainer = AnchorText(nlp, ['negative', 'positive'], use_unk_distribution=True)
    exp = explainer.explain_instance(test_text, wrapped_predict, threshold=0.95)
    exp.save_to_file("text_explanations/anchors_text_explanations/lime_test_data3.html", )
##############################################
# INITIALISING MODEL & DATA FOR IMDB DATASET #
##############################################
# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/text_data_1/IMDB Dataset.csv")
model_name = "xai_text_classification_data_1_lstm"
text_df, text_df['review'] = imdb_data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = imdb_data_processing(text_df['review'], text_df['sentiment'],0.3)
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 10, 64, model_name)
model = load_model("models/text_models/"+model_name+".h5")
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)
anchor_explainer(X_test, model, word_index, tokenizer)

##############################################
# INITIALISING MODEL & DATA FOR IMDB DATASET #
##############################################
news_df = dataset_2_build()
X_data, y_data = dataset_2_data_preprocessing(news_df)
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = dataset_2_data_processing(X_data, y_data ,0.3)
model_name = "xai_text_classification_data_2_lstm"
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 10, 32, model_name)
model = load_model("models/text_models/"+model_name+".h5")
anchor_explainer(X_test, model, word_index, tokenizer)
"""
#############################################################
# INITIALISING MODEL & DATA FOR AMAZON_YELP_TWITTER DATASET #
#############################################################
text_df_3 = pd.read_csv("datasets/text_data/text_data_3/amazon_yelp_twitter.csv")
model_name = "lime_xai_text_classification_data_3_lstm"
text_df_3.columns = ['sentiment', 'text']

text_df_3, text_df_3['text'] = dataset_3_data_preprocessing(text_df_3, text_df_3['text'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = dataset_3_data_processing(text_df_3['text'], text_df_3['sentiment'],0.3)
print(text_df_3.head())
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, 10, 64, model_name)
#model = load_model("models/text_models/"+model_name+".h5")
#lime_explainer(X_test, model, word_index, tokenizer)"""