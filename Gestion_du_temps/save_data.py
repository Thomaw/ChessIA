from pyautogui import screenshot
import cv2
import os
import glob


def save_clock_Lichess():
    """
    Sauvegarde des données pour l'entrainement du modèle de ML
    :param folder:
    :return:
    """

    # Constantes des positions des heures
    XMIN_HEURE_A = 352
    XMAX_HEURE_A = 424

    YMIN = 1295

    XMIN_HEURE_P = 756
    XMAX_HEURE_P = 832

    XPOSITION_M_OR_H = 30
    YPOSITION_M_OR_H = 250

    XRED = 4
    YRED = 4

    screenshot().save(r'Data_Number\ ' + str(1) + '.png')
    img = cv2.imread(r'Data_Number\ ' + str(1) + '.png')

    crop_img1 = img[XMIN_HEURE_A:XMAX_HEURE_A, YMIN:1557]  # Nombre avec des heures

    (r, g, b) = crop_img1[XPOSITION_M_OR_H, YPOSITION_M_OR_H]  # Vérifier que le nombre est à 4 ou 6 lettres
    if r == 233 and g == 235 and b == 237:
        crop_img1 = img[XMIN_HEURE_A:XMAX_HEURE_A, YMIN:1519]

    (r, g, b) = crop_img1[XRED, YRED]  # Couleur pour les 10 dernières secondes
    if r == 153 and g == 153 and b == 230:
        crop_img1 = img[XMIN_HEURE_A:XMAX_HEURE_A, YMIN:1565]  # Temps sous les 10 secondes

    crop_img2 = img[XMIN_HEURE_P:XMAX_HEURE_P, YMIN:1557]  # Nombre avec des heures

    (r, g, b) = crop_img2[XPOSITION_M_OR_H, YPOSITION_M_OR_H]  # Vérifier que le nombre est à 4 ou 6 lettres
    if r == 233 and g == 235 and b == 237:
        crop_img2 = img[XMIN_HEURE_P:XMAX_HEURE_P, YMIN:1519]

    (r, g, b) = crop_img2[XRED, YRED]  # Couleur pour les 10 dernières secondes
    if r == 153 and g == 153 and b == 230:
        crop_img2 = img[XMIN_HEURE_P:XMAX_HEURE_P, YMIN:1565]  # Temps sous les 10 secondes

    return [crop_img1, crop_img2]


def delete_residual_image(folder):
    """
     Supprime toutes les images d'un dossier
    :param folder:
    :return:
    """

    for png_file in glob.glob(folder + '\ ' + '*.png'):
        os.remove(png_file)
