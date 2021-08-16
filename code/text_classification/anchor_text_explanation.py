import random

import os
import os.path

import numpy as np
import spacy
from anchor.anchor_text import AnchorText
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

from text_classification_core.dataset_processing.text_dataset_1_processing import get_dataset_1
from text_classification_core.dataset_processing.text_dataset_2_processing import get_dataset_2
from text_classification_core.dataset_processing.text_dataset_3_processing import get_dataset_3


####################
# ANCHOR EXPLAINER #
####################
def anchor_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path):
    # Creating a reverse dictionary
    reverse_word_map = dict(map(reversed, word_index.items()))
    # Function takes a tokenized sentence and returns the words
    def sequence_to_text(list_of_indices):
        # Looking up words in dictionary
        words = [str(reverse_word_map.get(word)) for word in list_of_indices]
        return words
    my_texts = np.array(list(map(sequence_to_text, X_test)))

    def wrapped_predict(strings):
        cnn_rep = tokenizer.texts_to_sequences(strings)
        text_data = pad_sequences(cnn_rep, maxlen=max_sequence_length)
        prediction = model.predict(text_data)
        predicted_class = np.where(prediction > 0.5, 1,0)[0]
        return predicted_class

    nlp = spacy.load('en_core_web_sm')
    explainer = AnchorText(nlp, class_names, use_unk_distribution=True)

    random_indexes = random.sample(range(1,len(X_test)),3)
    for index in random_indexes:
        print("Creating prediction...")
        test_text = ' '.join(my_texts[index])
        exp = explainer.explain_instance(test_text, wrapped_predict, threshold=0.95)
        exp.save_to_file(save_path+'explanation_'+str(index)+'.html')
        print("Prediction saved...")

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
anchor_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.33) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
anchor_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative, positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/anchors_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.3, 30000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
anchor_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.
