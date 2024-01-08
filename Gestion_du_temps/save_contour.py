import cv2 as cv
import numpy as np


def _adaptiveThreshold_debug(file):
    """
    Utilise plusieurs masques pour obtenir une image facile d'utilisation

    :param file:
    :return:
    """

    img = cv.medianBlur(file, 5)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img, th3


def save_individual_number(file, folder_enregistrement):
    """
    Permet de sauvegarer dans un dossier les nombres grâce à la méthode des contours

    :param file:
    :param folder_enregistrement:
    :return:
    """

    area_list = list()
    ww = 1

    img_gray = cv.cvtColor(file, cv.COLOR_BGR2GRAY)
    img, th3 = _adaptiveThreshold_debug(img_gray)
    contours, _ = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)

        if x != 0 and y != 0 and w > 10 and h > 25 and 5000 > cv.contourArea(c) > 30:
            area_list.append([x, y, w, h])

    """
    Il existe des inclusions de contours (exmple le 0 ou le 4 reconnait deux contours pour chacun)
    Un contour i est inclus dans un contour j si:
    xi > xj, yi > yj, wi < wj, hi < hj, xi + wi < xj + wj, yi + hi < yj + hj
    """

    if img.shape[1] > 235:
        area_list = _contours_simplifications(len(area_list), 6, area_list)
    else:
        area_list = _contours_simplifications(len(area_list), 4, area_list)

    for c in area_list:
        cv.imwrite(folder_enregistrement + '\ ' + str(ww) + '.png', img[c[1]:c[1] + c[3], c[0]:c[0] + c[2]])
        ww += 1


def _contours_simplifications(u, min_u, area_list): # O seconde de calcul
    while u > min_u:
        for b in area_list:
            for a in area_list:
                if a[0] > b[0] and a[1] > b[1] and a[2] < b[2] and a[3] < b[3] \
                        and a[0] + a[2] < b[0] + b[2] and a[1] + a[3] < b[1] + b[3]:
                    area_list.remove(a)
                    u -= 1
                    break

    area_list = sorted(area_list, key=lambda x: x[0])  # reorder area_list en ordre décroissant (pour les x)
    return area_list
