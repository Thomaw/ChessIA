from time import *
import AI_function
import save_data
import save_contour
import glob
import numpy as np


def _list_to_clock(clock_list):
    """
    Fonction permettant de convertir une liste en temps sur Lichess
    :param clock_list:
    :return:
    """

    u = len(clock_list)
    clock_list.reverse()

    if u % 2 == 1:
        time_second = 0.1 * clock_list[0] + clock_list[1] + \
                      10 * clock_list[2] + (clock_list[3] + 10 * clock_list[4]) * 60

    else:
        # Le temps est écrit de la facon suivant : 1h45min24s = [4, 2, 5, 4, 1, 0]
        time_second = clock_list[0] + 10 * clock_list[1]
        if u > 2:
            time_second += (clock_list[2] + 10 * clock_list[3]) * 60
        if u > 4:
            time_second += (clock_list[4] + 10 * clock_list[5]) * 3600

        print(strftime('%H %M %S', gmtime(time_second)))
    return time_second


def Players_Time(json_file, h5_file):
    """
    Fonction permettant de récuprérer les temps des deux joueurs dans une liste
    :param json_file:
    :param h5_file:
    :return:
    """

    # Partie 0 :  Chargement du modèle de ML
    loaded_model = AI_function.load_CNN_model(json_file, h5_file) #~1.16 (total ~7.4)

    # Partie 1: extraction des temps
    temps = save_data.save_clock_Lichess()
    time_list = []

    # Partie 2 : Extraction des nombres d'une image + Conversion image/texte
    for file in temps:
        save_contour.save_individual_number(file, 'Nombre')
        clock_list = []

        for safe_contour in glob.glob('Nombre' + '\ ' + '*.png'):
            x_train = AI_function.preprocessing_image_classification(safe_contour)
            # start_time = time()
            prediction_class = np.argmax(loaded_model.predict(x_train), axis=-1) # équivalent à : prediction_class = loaded_model.predict_classes(x_train)
            # print("time elapsed: {:.2f}s".format(time() - start_time))

            clock_list.append(prediction_class[0])

        time_list.append(_list_to_clock(clock_list))
        save_data.delete_residual_image('Nombre')

    return time_list


def theoretical_time(k, alpha):
    return alpha * (59.3 + (72830-2330*k)/(2644+k*(10+k)))
