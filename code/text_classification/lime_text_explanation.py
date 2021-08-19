import random
import matplotlib.pyplot as plt
import keras.backend
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

from text_classification_core.dataset_processing.text_dataset_1_processing import get_dataset_1
from text_classification_core.dataset_processing.text_dataset_2_processing import get_dataset_2
from text_classification_core.dataset_processing.text_dataset_3_processing import get_dataset_3
from text_classification_core.text_pertubation import find_synonyms_and_perturb


#######################
# LIME TEXT EXPLAINER #
#######################
def lime_text_explainer(X_test_encoded, model, word_index, tokenizer, max_len, class_names, save_path):
    
    # Creating a dictionary of words to tokens.
    reverse_word_map = dict(map(reversed, word_index.items()))

    # Function which takes a tokenized sentence and returns the words.
    def sequence_to_text(list_of_indices):
        # Iterates through the sequence and returns the associated words.
        words = [str(reverse_word_map.get(word)) for word in list_of_indices]
        # Returned words
        return words
    # Calling the sequence_to_text method and applying it to all X_test data.
    my_texts = np.array(list(map(sequence_to_text, X_test_encoded)))
    
    # The LimeTextExplainer required a wrapped prediction function.
    def wrapped_predict(strings):
        # Converting string to explain back to a tokenized sequence. 
        cnn_rep = tokenizer.texts_to_sequences(strings)
        # Applying the original padding to the 
        text_data = pad_sequences(cnn_rep, maxlen=max_len)
        pred_list = model.predict(text_data)
        pred_list_final = []
        for index in range(len(pred_list)):
            prediction = pred_list[index][0]
            if prediction > 0.5:
                pred_list_final.append(np.insert(pred_list[index], 0, (1-prediction)))
            else: 
                pred_list_final.append(np.insert(pred_list[index], 1, (prediction*-1)))
        pred_list_final = np.array(pred_list_final)
        return pred_list_final

    # Initialising the LimeTextExplainer
    explainer = LimeTextExplainer(class_names=class_names)

    random_indexes = random.sample(range(1,len(X_test)),4)
    for index in random_indexes:
        test_text = ' '.join(my_texts[index])
        peturbed_string = find_synonyms_and_perturb(my_texts[index])
        peturbed_string = ' '.join(peturbed_string)
        peturbed_string_1 = 'this film is beaming face with smiling eyes I just watched it'

        exp_peturb = explainer.explain_instance(peturbed_string_1, wrapped_predict, num_features=6, labels=(1,))
        exp_peturb.save_to_file(save_path+'explanation_peturb'+str(index)+'.html')
        
        exp = explainer.explain_instance(test_text, wrapped_predict, num_features=6, labels=(1,))
        exp.save_to_file(save_path+'explanation_'+str(index)+'.html')

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_1 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_1/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_1(0.3) # Processing the data preparing for training
model_name = "xai_text_classification_data_1_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_2 #
############################################
class_names = ['fake', 'real'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_2/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_2(0.3) # Processing the data preparing for training
model_name = "xai_text_classification_data_2_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.

############################################
# GENERATING EXPLAINATIONS FOR TEXT_DATA_3 #
############################################
class_names = ['negative', 'positive'] # The class names of the dataset, the order is important.
save_path = 'text_explanations/lime_text_explanations/text_data_3/' # The filepath to save the related explanations.
X_train, X_test, y_train, y_test, max_sequence_length, vocab_size, word_index, tokenizer = get_dataset_3(0.3, 40000) # Processing the data preparing for training
model_name = "xai_text_classification_data_3_lstm" # Name of associated model
model = load_model("models/text_models/"+model_name+".h5") # Loading the trained model.
lime_text_explainer(X_test, model, word_index, tokenizer, max_sequence_length, class_names, save_path) # Generating a explanation.
