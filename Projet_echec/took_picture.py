import pyautogui
import time
import cv2 as cv
import numpy as np
import Constantes_values


############################################### TOUT EST A REFAIRE A CAUSE DU CHANGEMENT D'ORDI  ###############################################

def took_picture():
    """Réalise un screenshot du plateau de jeu"""

    time.sleep(1)

    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'Data_Pictures\test.jpg')


def read_image():
    """Redimensionne l'image pour garder uniquement le plateau de jeu"""

    img = cv.imread(r'Data_Pictures\test.jpg')  # Récupération de l'image
    crop_img = img[205:972, 510:1277]

    return crop_img


def obtain_a_square(img, i, j):
    """Fonction pour récupérer un carré du plateau de jeu"""
    carre_image = img[round(Constantes_values.largeur_carre * i):round(Constantes_values.largeur_carre * (i + 1)),
                  round(Constantes_values.largeur_carre * j):round(Constantes_values.largeur_carre * (j + 1))]

    return carre_image


def save_a_square(img, i, j):
    """Sauvegarde l'image du plateau de jeu"""
    existing_piece = False

    carre_image = obtain_a_square(img, i, j)

    (b, g, r) = carre_image[round(Constantes_values.largeur_carre / 2), round(Constantes_values.largeur_carre / 2)]

    # print(b,g,r)

    if r != 181 and r != 240 and r != 170 and r != 205:
        cv.imwrite('Data_Pictures\ ' + str(8 * i + j) + '.png', carre_image)
        existing_piece = True

    return carre_image, existing_piece


# Positionnement des carrées:
# Un carré fait 65.375 de coté
# La position du carré (i,j) est (350+math.ceil(largeur_carre*i), 162+math.ceil(largeur_carre*j)) pour
# son extrmité haut gauche:
# (350+math.ceil(largeur_carre*i)+largeur_carre/2 , 162+math.ceil(largeur_carre*j)+largeur_carre/2)
def position_carre(i, j):
    """Renvoie la position centrale d'un carré de jeu"""
    return 508 + round(Constantes_values.largeur_carre * i + Constantes_values.largeur_carre / 2), \
           201 + round(Constantes_values.largeur_carre * j + Constantes_values.largeur_carre / 2)


def calcPercentage(msk):
    ''' returns the percentage of color in a binary image '''
    height, width = msk.shape[:2]
    num_pixels = height * width
    count_white = cv.countNonZero(msk)
    percent_white = (count_white / num_pixels) * 100
    percent_white = round(percent_white, 2)
    return percent_white


def piece_color(file):
    """Fonction donnant la couleur de la pièce à partir du fichier d'une image"""
    img = cv.imread(file, cv.IMREAD_COLOR)  # Lecture de l'image en couleurs

    return piece_color_from_image(img)


def piece_color_from_image(img):
    """Fonction donnant la couleur de la pièce à partir d'une image"""
    White_lower_bound = np.array([252, 252, 252])
    White_upper_bound = np.array([255, 255, 255])

    Black_lower_bound = np.array([0, 0, 0])
    Black_upper_bound = np.array([3, 3, 3])

    White_msk = cv.inRange(img, White_lower_bound, White_upper_bound)
    Black_msk = cv.inRange(img, Black_lower_bound, Black_upper_bound)

    percent_white = calcPercentage(White_msk)
    percent_black = calcPercentage(Black_msk)

    if percent_white > percent_black:
        result = True
    else:
        result = False

    return result


# semble fonctionner mais à vérifier
def find_movement():
    """Fonction permettant de récupérer le mouvement de la pièce sur lichess"""
    crop_img = read_image()
    move_list = [0, 0]
    u = 0

    for i in range(0, 8):
        for j in range(0, 8):
            (r, g, b) = crop_img[round(Constantes_values.largeur_carre * i + Constantes_values.largeur_carre * 9 / 10),
                                 round(Constantes_values.largeur_carre * j + Constantes_values.largeur_carre * 9 / 10)]

            # print('(i,j) :', (i,j), (r, g, b))

            condition = (100<=r<=122 and 202<=g<=212 and 201<=b<=212) or (47<=r<=67 and 155<=g<=165 and 165<=b<=175)

            if condition:
                u += 1

                (r, g, b) = crop_img[
                    round(Constantes_values.largeur_carre * i + Constantes_values.largeur_carre / 2),
                    round(Constantes_values.largeur_carre * j + Constantes_values.largeur_carre / 2)]

                # print('Trouvée : (i,j) :', (i, j), (r, g, b))
                condition = (100 <= r <= 122 and 202 <= g <= 212 and 201 <= b <= 212) or (47 <= r <= 67 and 155 <= g <= 165 and 165 <= b <= 175)

                if condition:
                    #print(i, j, r, g, b, "debut du déplacement")
                    move_list[0] = 8 * i + j

                else:
                    # print(i, j, r, g, b, "Fin du déplacement")
                    move_list[1] = 8 * i + j

                if u == 2:
                    break

    # pas sur de ce if
    if u != 2:
        move_list = find_rock_movement()

    print(move_list)
    return move_list



# NE MARCHE PAS A CAUSE DE CES PUTAINS DE COULEURS
def find_rock_movement():
    """Fonction permettant de récupérer le rock sur lichess"""
    crop_img = read_image()
    move_list = [0, 0]
    u = 0

    # Pour chaque pièce de l'échiquier
    for i in range(0, 8):
        for j in range(0, 8):


            # Regarde la couleur de la pièce
            (r, g, b) = crop_img[round(Constantes_values.largeur_carre * i + Constantes_values.largeur_carre * 9 / 10),
                                 round(Constantes_values.largeur_carre * j + Constantes_values.largeur_carre * 9 / 10)]

            condition = (100 <= r <= 122 and 202 <= g <= 212 and 201 <= b <= 212) or (47 <= r <= 67 and 155 <= g <= 165 and 165 <= b <= 175)

            # Récupère la couleur verte du plateau
            # Pour un rock, il y a deux images vertes sans pièces dessus
            if condition:

                (r, g, b) = crop_img[
                    round(Constantes_values.largeur_carre * i + Constantes_values.largeur_carre / 2),
                    round(Constantes_values.largeur_carre * j + Constantes_values.largeur_carre / 2)]

                condition = (100 <= r <= 122 and 202 <= g <= 212 and 201 <= b <= 212) or (47 <= r <= 67 and 155 <= g <= 165 and 165 <= b <= 175)

                if condition:
                    move_list[u] = 8 * i + j
                    u += 1

                if u == 2:
                    break

        print(move_list)
    return move_list

