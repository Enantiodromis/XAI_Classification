import keras.backend as backend
from keras.backend.cntk_backend import reverse
import numpy as np
import shap
import pandas as pd
from keras.models import load_model
from text_classification_core import lstm_model, plot_accuracy_loss
from imdb_data_clean_process import data_preprocessing, data_processing
from keras.preprocessing.text import Tokenizer
import cntk as C

##################
# SHAP EXPLAINER #
##################
def shap_explainer(X_train_not_encoded, X_test_not_encoded, X_train_encoded, X_test_encoded, model, word_index, y_test_cpy, tokenizer):
    print("Loading SHAP Explanation...")
    """y_test_cpy = np.array(y_test_cpy).flatten()
    background = X_train_encoded[:100]
    session = backend.get_session()
    explainer = shap.DeepExplainer(model, background, session)
    
    num_explanation = 10  
    shap_values = explainer.shap_values(X_test_encoded[:num_explanation])
    print(shap_values)"""
    #print(tokenizer.sequences_to_texts(X_test_encoded[1]))
    #shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test_encoded[0])

    """import shap

    # we use the first 100 training examples as our background dataset to integrate over
    explainer = shap.DeepExplainer(model, X_train_encoded[:100])

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(X_train_encoded[:10])
    # init the JS visualization code
    shap.initjs()

    # transform the indexes to words
    import numpy as np
    reverse_word_map = dict(map(reversed, word_index.items()))
    #print(shap_values)
    # plot the explanation of the first prediction
    # Note the model is "multi-output" because it is rank-2 but only has one column
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], reverse_word_map[0])"""

    # We use the first 100 training examples as our background dataset to integrate over
    # Initialising background for DeepExplainer
    background = X_train_encoded[:100]
    #print(background)
    explainer = shap.DeepExplainer(model, background)

    # Explain the first 10 predicitions
    # Explaining each predicition requires 2*background dataset size runs
    shap_values = explainer.shap_values(X_test_encoded[:1])
    print(shap_values[0])
    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    print("MADE IT HERE")
    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(letter) for letter in list_of_indices]
        return words

    # Creating texts 
    #np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])
    my_texts = np.array(list(map(sequence_to_text, X_test_encoded[:10])))
    print(my_texts)

    shap.initjs()
    folder = shap.force_plot(explainer.expected_value[0], shap_values[0][0], my_texts[0])
    file ='force_plot.html'
    shap.save_html(file, folder)

# Reading in the csv, containing the the csv data
text_df = pd.read_csv("datasets/text_data/IMDB Dataset.csv")
epoch_num = 3
model_name = "lstm_Imdb_dataset"
text_df, text_df['review'] = data_preprocessing(text_df, text_df['review'])
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, X_train_cpy, X_test_cpy, word_index, y_test_cpy, tokenizer = data_processing(text_df['review'], text_df['sentiment'],0.3)
#history, model = lstm_model(vocab_size, X_train, y_train, X_test, y_test, epoch_num, 64, model_name)
model = load_model("models/"+model_name+".h5")
#model = load_model("models/LSTM_11.h5") # Loading the saved model
#plot_accuracy_loss(history, epoch_num)
shap_explainer(X_train_cpy, X_test_cpy, X_train, X_test, model, word_index, y_test_cpy, tokenizer)
