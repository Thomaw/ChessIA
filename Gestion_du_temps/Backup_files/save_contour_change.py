import cv2 as cv
from matplotlib import pyplot as plt


def adaptiveThreshold_debug(file, plt_show=False):
    """
    Utilise plusieurs masques pour obtenir une image facile d'utilisation

    :param file:
    :param plt_show:
    :return:
    """
    img = cv.imread(file, 0)
    img = cv.medianBlur(img, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    if plt_show:
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    return img, th1, th2, th3


def save_individual_number(file, folder_enregistrement):
    """
    Permet de sauvegarer dans un dossier les nombres grâce à la méthode des contours

    :param file:
    :param folder_enregistrement:
    :return:
    """
    img = cv.imread(file, 0)
    dimension = img.shape
    print(dimension)

    ww = 1

    img, th1, th2, th3 = adaptiveThreshold_debug(file)
    contours, hierarchy = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    area_list = list()
    u = 0

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        # print(x, y, w, h)

        # if x != 0 and y != 0 and w < 190 and h < 35 and 5000 > cv.contourArea(c) > 30: # Bon pour les nombres supérieures à 1h
        if x != 0 and y != 0 and w > 10 and h > 25 and 5000 > cv.contourArea(c) > 30:
            u += 1
            area_list.append([x, y, w, h])
            # print("Contours saved : ", x, y, w, h, cv.contourArea(c))

    """
    Il existe des inclusions de contours (exmple le 0 ou le 4 reconnait deux contours pour chacun)
    Un contour i est inclus dans un contour j si:
    xi > xj, yi > yj, wi < wj, hi < hj, xi + wi < xj + wj, yi + hi < yj + hj
    """

    if dimension[1] > 235:
        area_list = contours_simplifications(u, 6, area_list)
    else:
        area_list = contours_simplifications(u, 4, area_list)

    # print(len(area_list))

    with_contours = cv.drawContours(img, contours, -1, (255, 255, 255), 1)
    area_list = sorted(area_list, key=lambda x: x[0])  # reorder area_list en ordre décroissant (pour les x)

    for c in area_list:
        cv.rectangle(with_contours, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (255, 255, 255), 1)

        temp_imag = img[c[1]:c[1] + c[3], c[0]:c[0] + c[2]]
        cv.imwrite(folder_enregistrement + '\ ' + str(ww) + '.png', temp_imag)
        print(folder_enregistrement + '\ ' + str(ww) + '.png', ' Position : ', c[0], c[1], c[2], c[3])
        ww += 1

    cv.imshow("contours", with_contours)
    cv.waitKey(0)


def contours_simplifications(u, min_u, area_list):
    while u > min_u:
        for b in area_list:
            for a in area_list:
                if a[0] > b[0] and a[1] > b[1] and a[2] < b[2] and a[3] < b[3] \
                        and a[0] + a[2] < b[0] + b[2] and a[1] + a[3] < b[1] + b[3]:
                    area_list.remove(a)
                    u = len(area_list)
                    break

    return area_list


# adaptiveThreshold_debug(r'a_supprimer\ 1.png', True)
save_individual_number(r'a_supprimer\ 2.png', 'Backup_files')
