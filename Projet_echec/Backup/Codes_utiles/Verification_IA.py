import AI_function
import numpy as np
import glob


labels = ['Pion', 'Cavalier', 'Fou', 'Tour', 'Dame', 'Roi']


def IA_check_function(json_file, h5_file):
    """
    Fonction permettant de récuprérer les temps des deux joueurs dans une liste
    :param json_file:
    :param h5_file:
    :return:
    """
    # Partie 0 :  Chargement du modèle de ML
    loaded_model = AI_function.load_CNN_model(json_file, h5_file)  # ~1.16 (total ~7.4)

    # Partie 2 : Extraction des nombres d'une image + Conversion image/texte
    for file in glob.glob(r'AI_model\Test_image\Dame' + '\\' + '*.png'):
        x_train = AI_function.preprocessing_image_classification(file)
        prediction_class = np.argmax(loaded_model.predict(x_train), axis=-1)
        print(labels[prediction_class[0]])


IA_check_function('model.json', 'model.h5')
