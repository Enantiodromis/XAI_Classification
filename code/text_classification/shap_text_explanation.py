import os
import os.path
import random

import cntk as C
from matplotlib.pyplot import show
import numpy as np
import pandas as pd
import shap
from keras.backend.cntk_backend import reverse
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from text_classification_core.dataset_processing.text_dataset_1_processing import get_dataset_1
from text_classification_core.dataset_processing.text_dataset_2_processing import get_dataset_2
from text_classification_core.dataset_processing.text_dataset_3_processing import get_dataset_3


##################
# SHAP EXPLAINER #
##################
def shap_explainer(X_test, model, word_index, save_path):
    print("Loading SHAP Explanation...")
    # We use the first 100 training examples as our background dataset to integrate over
    # Initialising background for DeepExplainer
    background = X_test[:100]

    explainer = shap.DeepExplainer(model, background)

    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [reverse_word_map.get(word) for word in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test)))

    num_explanations = 3
    # Explain the first 10 predicitions
    # Explaining each predicition requires 2*background dataset size runs
    shap_values = explainer.shap_values(X_test[:3])

    for index in range(num_explanations):
        shap.initjs()
        shap.save_html(save_path+'explanation_'+str(index)+'.html', shap.force_plot(explainer.expected_value[0], shap_values[0][index], my_texts[index]))

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_test, model, word_index, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_test, model, word_index, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative, positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/shap_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.3, 30000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
shap_explainer(X_test, model, word_index, save_path) # Generating a explanation.
