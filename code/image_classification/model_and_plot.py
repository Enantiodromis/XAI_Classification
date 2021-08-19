import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


####################
# DATASET CREATION #
####################
def binary_dataset_creation(batch_size, img_height, img_width, from_dataframe, need_train_test_split, dataframe = None, file_path = None):
    if need_train_test_split == False:
        data_generator = ImageDataGenerator()
        if from_dataframe == True:
            generator = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True
            )
        else:
            generator = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                shuffle=True   
            )
        return generator
    else: 
        data_generator = ImageDataGenerator(validation_split=0.2)
        if from_dataframe == True:
            generator_1 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True
            )
            generator_2 = data_generator.flow_from_dataframe(
                dataframe = dataframe,
                x_col= 'path',
                y_col= 'label_str',
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True
            )
        else:
            generator_1 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="training",
                shuffle=True   
            )
            generator_2 = data_generator.flow_from_directory(
                directory = file_path,
                batch_size= batch_size,
                target_size= (img_height,img_width),
                class_mode= 'binary',
                subset="validation",
                shuffle=True   
            )
        return generator_1, generator_2

#######################################
# BUILD, TRAIN AND EVALUATE THE MODEL #
#######################################
def img_classification_model(train_generator, test_generator, number_epochs, model_name):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer= RMSprop(lr=1e-4),
                metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=(len(train_generator) / train_generator.batch_size),
        epochs=number_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
    )
    np.save('model_history/'+model_name+'.npy',history.history)
    model.save('models/image_models/'+model_name+'.h5') 

    return history, model

def plot_accuracy_loss(model_history, number_epochs, model_name):
    x_list = []
    x_list.extend(range(number_epochs))

    plt.figure(2,figsize=(15,4))
    plt.plot(model_history['acc'])
    plt.plot(model_history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_acc.jpg")

    plt.figure(3,figsize=(15,4))
    plt.plot(model_history['loss'], color='green')
    plt.plot(model_history['val_loss'], color='red')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(x_list)
    plt.tight_layout()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("data_plots/"+model_name+"_loss.jpg")
