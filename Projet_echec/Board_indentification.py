import time
import cv2
import numpy as np
import took_picture
import glob
import os
import AI_function
import IA_database


def board_pieces_detection():
    """Détecte les pièces lors des screenshots"""

    """ Supprime toutes les images du dossier """
    files = glob.glob('Data_Pictures/*')
    for f in files:
        os.remove(f)

    # start_time = time.time()
    loaded_model = AI_function.load_CNN_model('model.json', 'model.h5')
    # print('Temps de chargement du model : ', time.time() - start_time)

    take_picture = True
    if take_picture:
        took_picture.took_picture()

    crop_img = took_picture.read_image()
    pieces_position = np.zeros(64, np.int32)

    '''
    1 :  Pion blanc              7 :  Pion noir
    2 :  Cavalier blanc          8 :  Cavalier noir
    3 :  Fou blanc               9 :  Fou noir
    4 :  Tour blanche            10 : Tour noire
    5 :  Dame blanche            11 : Dame noire
    6 :  Roi blanc               12 : Roi noir
    '''

    for i in range(0, 8):
        for j in range(0, 8):
            carre_image, existing_image = took_picture.save_a_square(crop_img, i,
                                                                     j)  # Sauvegarde les images ayant une pièce

            if existing_image:  # Si une pièce existe
                piece_color = took_picture.piece_color_from_image(carre_image)  # Récupère la couleur de la pièce
                image = cv2.imread('Data_Pictures\ ' + str(8 * i + j) + '.png', 0)

                th3 = IA_database.thresholding(image)
                cv2.imwrite('AI_Pictures\ ' + str(8 * i + j) + '.png', th3)

                x_train = AI_function.preprocessing_image_classification('AI_Pictures\ ' + str(8 * i + j) + '.png')
                prediction_class = np.argmax(loaded_model.predict(x_train), axis=-1)

                if piece_color:
                    pieces_position[8 * i + j] = prediction_class[0]+1
                else:
                    pieces_position[8 * i + j] = prediction_class[0] + 7


    affichage_echiquier(pieces_position)
    # time.sleep(60)
    # os.remove(r'Data_Pictures\test.png')
    return pieces_position


def affichage_echiquier(pieces_position):
    """Affichage de l'échiquier dans la commande windows"""
    for i in range(0, 8):
        print('| ' + str(pieces_position[8 * i]) + ' | ' + str(pieces_position[8 * i + 1]) + ' | ' + str(
            pieces_position[8 * i + 2]) + ' | ' + str(pieces_position[8 * i + 3]) + ' | ' + str(
            pieces_position[8 * i + 4]) + ' | ' + str(pieces_position[8 * i + 5]) + ' | ' + str(
            pieces_position[8 * i + 6]) + ' | ' + str(pieces_position[8 * i + 7]) + ' | ')

    print("")
#
# board_pieces_detection()
