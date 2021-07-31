import keras.backend as backend
from keras_preprocessing.sequence import pad_sequences
#from keras.backend.cntk_backend import reverse
import numpy as np
#import shap
import pandas as pd
from keras.models import load_model
from text_classification_core import lstm_model, plot_accuracy_loss
from imdb_data_clean_process import data_preprocessing, data_processing
#import cntk as C
import webbrowser
from lime.lime_text import LimeTextExplainer
from pprint import pprint
import json

##################
# LIME EXPLAINER #
##################
def lime_explainer(X_train_not_encoded, X_test_not_encoded, X_train_encoded, X_test_encoded, model, word_index, y_test_cpy, tokenizer):

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test_encoded)))
    
    explainer = LimeTextExplainer(class_names=['negative','positive'])
    def wrapped_predict(strings):
        cnn_rep = tokenizer.texts_to_sequences(strings)
        text_data = pad_sequences(cnn_rep, maxlen=30)
        return model.predict(text_data)

    test_text = ' '.join(my_texts[4])
    exp = explainer.explain_instance(test_text, wrapped_predict, num_features=10, top_labels=2) 
    
    print(X_test_not_encoded)
    url = exp.as_html()
    with open("test.html", "w", encoding="utf-8") as f:
        f.write(url)
    import webbrowser
    filepath = "test.html"
    webbrowser.open(filepath,new=2)

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")
epoch_num = 3
model_name = "lstm_Imdb_dataset"
text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = data_processing(text_df['review'], text_df['sentiment'],0.3)
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, epoch_num, 64, model_name)
model = load_model("models/"+model_name+".h5")
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)
lime_explainer(X_train_cpy, X_test_cpy, X_train, X_test, model, word_index, y_test_cpy, tokenizer)