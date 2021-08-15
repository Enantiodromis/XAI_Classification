import keras.backend as backend
from keras.backend.cntk_backend import reverse
import numpy as np
import shap
import pandas as pd
from keras.models import load_model
from text_classification_core import lstm_model, plot_accuracy_loss
from dataset_processing.text_dataset_1_processing import imdb_data_preprocessing, imdb_data_processing
from keras.preprocessing.text import Tokenizer
import cntk as C
import webbrowser

##################
# SHAP EXPLAINER #
##################
def shap_explainer(X_train_not_encoded, X_test_not_encoded, X_train_encoded, X_test_encoded, model, word_index, y_test_cpy, tokenizer):
    print("Loading SHAP Explanation...")
    # We use the first 100 training examples as our background dataset to integrate over
    # Initialising background for DeepExplainer
    background = X_train_encoded[:100]
    #print(background)
    explainer = shap.DeepExplainer(model, background)

    # Explain the first 10 predicitions
    # Explaining each predicition requires 2*background dataset size runs
    shap_values = explainer.shap_values(X_test_encoded[:1])

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words

    # Creating texts 
    #np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])
    my_texts = np.array(list(map(sequence_to_text, X_test_encoded)))
    shap.initjs()
    shap.save_html("text_explanations/shap_text_explanations/test.html",shap.force_plot(explainer.expected_value[0], shap_values[0][0], my_texts[0]))

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/text_data_1/IMDB Dataset.csv")
epoch_num = 3
model_name = "xai_text_classification_data_1_lstm"
text_df, text_df['review'] = imdb_data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = imdb_data_processing(text_df['review'], text_df['sentiment'],0.3)
# history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, epoch_num, 64, model_name)
model = load_model("models/text_models/"+model_name+".h5")
#history=np.load("model_history/"+model_name+".npy",allow_pickle='TRUE').item() # Loading the training history to be used for plotting
#plot_accuracy_loss(history, epoch_num, model_name)
shap_explainer(X_train_cpy, X_test_cpy, X_train, X_test, model, word_index, y_test_cpy, tokenizer)
