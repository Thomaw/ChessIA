import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.losses import SparseCategoricalCrossentropy

labels = ['Pion', 'Cavalier', 'Fou', 'Tour', 'Dame', 'Roi']
img_size = 128


def get_data(data_dir):
    """
    Fonction pour obtenir les données qui servent d'entrainement et de test
    :param data_dir:
    :return:
    """

    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size))   # Reshaping images to preferred size
            data.append([resized_arr, class_num])

    return np.array(data, dtype=object)


def preprocessing_data(train_value, values):
    """
    Fonction de preprocessing des données
    :param train_value:
    :param values:
    :return:
    """

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train_value:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in values:
        x_val.append(feature)
        y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val


def model_CNN():
    """
    Création du modèle de ML
    :return:
    """

    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(6, activation="softmax"))

    model.summary()

    opt = Adam(lr=0.000001)
    model.compile(optimizer=opt, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


def load_CNN_model(json_fill, h5_file):
    """
    Permet de charger le modèle CNN
    :param json_fill:
    :param h5_file:
    :return:
    """

    if os.path.isfile(json_fill) and os.path.isfile(h5_file):
        json_file = open(json_fill, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_file)

        return loaded_model

    else:
        print("Le modèle n'existe pas")
        return 0


def save_CNN_model(model, json_name, h5_name):
    """
    Permets de sauvegarder le model CNN entrainé
    :param model:
    :param json_name:
    :param h5_name:
    :return:
    """

    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(h5_name)
    print("Saved model to disk")


def preprocessing_image_classification(file):
    """
    Fonction pour le preprocessing lors d'une classification d'image
    :param file:
    :return:
    """
    x_train = np.array([cv2.resize(cv2.imread(file)[..., ::-1], (img_size, img_size))]) / 255
    x_train.reshape(-1, img_size, img_size, 1)

    return x_train
