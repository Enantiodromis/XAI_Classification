###################################################################################
# 										  #
#		        XAI_CLASSIFICATION_MIXED_DATA README		          #
#										  #
###################################################################################
+=================================================================================+
|                                 # DEPENDENCIES #				  |
+=================================================================================+
| Due to the different implemetation stages of the XAI techniques used, there is a|
| large discrepency of package, python verions etc... needed to run the code.     |
|										  |	
| We recommended that you create four virtual environments to run this code.      |
|										  |
| The dependencies for each recommended environment:				  |
| The dependency files should be read in and the packages installed using pip.    |
+=================================================================================+
+---------------------------------------------------------------------------------+
|       Environment      | Python version |  Associated package dependency file   |
|                        |                |                                       |
+------------------------+----------------+---------------------------------------+
| anchor_image_explainer |      3.9.6     | anchor_image_package_dependencies.txt |
+------------------------+----------------+---------------------------------------+
| lime_image_explainer   |      3.9.5     | lime_image_package_dependencies.txt   |
+------------------------+----------------+---------------------------------------+
| shap_image_explainer   |      3.6.8     | shap_image_package_dependencies.txt   |
+------------------------+----------------+---------------------------------------+
| text_explainer         |     3.6.13     | text_package_dependencies.txt         |
+------------------------+----------------+---------------------------------------+

+=================================================================================+
|                                 #  DATASETS  #				  |
+=================================================================================+
| Six datasets are needed to run this project, the table below lists them with a  |
| link to where they can be installed.                                            |
|										  |	
| Once downloaded they should be unpacked and placed into the associated folder   |
| within the datasets folder.		    					  |
+=================================================================================+


+=================================================================================+
|                       # CLASS AND FUNCTION EXPLANATION #	                  |
+=================================================================================+
| We briefly explain the classes and the functions of our implemenation  	  |
|										  |
| For ease of explanation we explain image & text explanation code seperately.    |
| The classes are described in the two tables below with each instance in logical |
| order to run and generate explanations.					  |					  	
+=================================================================================+
+---------------------------------------------------------------------------------------------------------------------+
|                                  IMAGE EXPLANATIONS (FOLDER: image_classification)                                  |
+-------------------------------+------------------------------------+------------------------------------------------+
|              File             |              Function              |                     Purpose                    |
|                               |                                    |                                                |
+-------------------------------+------------------------------------+------------------------------------------------+
| image_dataset_1_processing.py | - get_dataset_1()                  | The functions, load, process and return the    |
| image_dataset_2_processing.py | - get_dataset_2()                  | image datasets ready for training and generat- |
| image_dataset_2_processing.py | - get_dataset_3()                  | ing image explanations.                        |
+-------------------------------+------------------------------------+------------------------------------------------+
| image_perturbation.py         | - image_perturbation()             | The image_perturbation function takes as input |
|                               |                                    | an image and returns a perturbed image.        |
+-------------------------------+------------------------------------+------------------------------------------------+
| model_and_plot.py             | - binary_dataset_creation()        | The binary_dataset_creation function loads the |
|                               | - img_classification_model()       | dataset passed to it using an ImageGenerator.  |
|                               | - plot_accruracy_loss_multiple()   |                                                |
|                               |                                    | The img_classification_model function calls    |
|                               |                                    | the training of the image classification model |
|                               |                                    |                                                |
|                               |                                    | The plot_accuracy_loss_multiple model is the   |
|                               |                                    | implementation which generates graphs of the   |
|                               |                                    | model's loss and accuracy.                     |
+-------------------------------+------------------------------------+------------------------------------------------+
| model_training.py             | N/A                                | Running this file, trains and saves all the    |
|                               |                                    | image classification models and plots their    |
|                               |                                    | accuracy and loss.                             |
+-------------------------------+------------------------------------+------------------------------------------------+
| anchor_image_explanation.py   | - extracting_anchors_explanation() | This method generates Anchor explanations from |
|                               |                                    | instances of the data passed to it.            |
+-------------------------------+------------------------------------+------------------------------------------------+
| lime_image_explanation.py     | - extracting_lime_explanation()    | This method generates LIME explanations from   |
|                               |                                    | instances of the data passed to it.            |
+-------------------------------+------------------------------------+------------------------------------------------+
| shap_image_explanation.py     | - extracting_shap_explanation()    | This method generates SHAP explanations from   |
|                               |                                    | instances of the data passed to it.            |
+-------------------------------+------------------------------------+------------------------------------------------+

+--------------------------------------------------------------------------------------------------------------------------+
|                                      TEXT EXPLANATIONS (FOLDER: Text_classification)                                     |
+-------------------------------+------------------------------------+-----------------------------------------------------+
|              File             |              Function              |                       Purpose                       |
|                               |                                    |                                                     |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| image_dataset_1_processing.py | - dataset_1_build()                | dataset_1_build - Loads in data from csv.           |
|                               | - get_1_preprocessing()            |                                                     |
|                               | - get_dataset_1()                  | get_1_preprocessing - Data cleaning by dropping     |
|                               |                                    | NaN values, converting emojis, regex and stopwords. |
|                               |                                    |                                                     |
|                               |                                    | get_dataset_1 - Binarizers and tokenizes data and   |
|                               |                                    | returns data and data features.                     |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| image_dataset_2_processing.py | - dataset_2_build()                | dataset_2_build - Loads in data from csv.           |
|                               | - get_2_preprocessing()            |                                                     |
|                               | - get_dataset_2()                  | get_2_preprocessing - Data cleaning by dropping     |
|                               |                                    | NaN values, converting emojis, regex and stopwords. |
|                               |                                    |                                                     |
|                               |                                    | get_dataset_2 - Binarizers and tokenizes data and   |
|                               |                                    | returns data and data features.                     |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| image_dataset_3_processing.py | - dataset_3_build()                | dataset_3_build - Loads in data from csv.           |
|                               | - get_3_preprocessing()            |                                                     |
|                               | - get_dataset_3()                  | get_3_preprocessing - Data cleaning by dropping     |
|                               |                                    | NaN values, regex and stopwords.                    |
|                               |                                    |                                                     |
|                               |                                    | get_dataset_3 - Binarizers and tokenizes data and   |
|                               |                                    | returns data and data features.                     |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| text_perturbation.py          | - text_perturbation()              | The text_perturbation function takes as input       |
|                               |                                    | a text instance and returns a perturbed text.       |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| model_and_plot.py             | - binary_dataset_creation()        | The binary_dataset_creation function loads the      |
|                               | - lstm_model()                     | dataset passed to it using a Generator.             |
|                               | - plot_accruracy_loss_multiple()   |                                                     |
|                               |                                    | The lstm_model function calls the training of the   |
|                               |                                    | text classification model                           |
|                               |                                    |                                                     |
|                               |                                    | The plot_accuracy_loss_multiple model is the        |
|                               |                                    | implementation which generates graphs of the        |
|                               |                                    | model's loss and accuracy.                          |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| model_training.py             | N/A                                | Running this file, trains and saves all the         |
|                               |                                    | image classification models and plots their         |
|                               |                                    | accuracy and loss.                                  |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| anchor_image_explanation.py   | - extracting_anchors_explanation() | This function generates Anchor explanations from    |
|                               |                                    | instances of the data passed to it.                 |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| lime_image_explanation.py     | - lime_text_explainer()            | This function generates LIME explanations from      |
|                               |                                    | instances of the data passed to it.                 |
+-------------------------------+------------------------------------+-----------------------------------------------------+
| shap_image_explanation.py     | - shap_explainer()                 | This function generates SHAP explanations from      |
|                               |                                    | instances of the data passed to it.                 |
+-------------------------------+------------------------------------+-----------------------------------------------------+





