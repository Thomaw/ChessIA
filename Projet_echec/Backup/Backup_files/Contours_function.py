import cv2
import took_picture
import numpy as np
import Constantes_values


def resize_picture(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dsize = (width, height)
    output = cv2.resize(img, dsize)

    return output


def Color_inversion(img):  # Read pixels and apply negative transformation
    for i in range(0, img.shape[0] - 1):
        for j in range(0, img.shape[1] - 1):
            img[i, j] = 255 - img[i, j]  # Get pixel value at (x,y) position of the image


def Caracteristic_white_contour(file):
    percent = 100
    piece_color = took_picture.piece_color(file)

    img2 = cv2.imread(file, cv2.IMREAD_COLOR)  # Lecture de l'image en couleurs
    img2 = resize_picture(img2, percent)

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Reading same image in another variable and converting to gray scale.
    img = resize_picture(img, percent)

    if not piece_color:  # Si la pièce est noire
        Color_inversion(img)  # img devient négative
        # cv2.imshow('Color_inversion', img)

    _, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)  # Converting image to a binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Detecting contours in image.

    xc = list()
    yc = list()
    contour_length = 0

    nb_contour = 0
    for cnt in contours:  # Going through every contours found in the image.
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # print(cv2.arcLength(cnt, True))
        if cv2.arcLength(cnt, True) > 5:
            contour_length += cv2.arcLength(cnt, True)
            nb_contour += 1

        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 1)

        n = approx.ravel()  # Used to flatted the array containing the co-ordinates of the vertices.
        i = 0

        for j in n:
            if i % 2 == 0:
                xc.append(n[i])
                yc.append(n[i + 1])
            i += 1

    return [xc, yc, nb_contour, contour_length]


def contour_white_piece_detection(file):
    # contour_length_list = list()
    # nb_contour_list = list()

    xc, yc, nb_contour, contour_length = Caracteristic_white_contour(file)
    # contour_length_list.append(contour_length)
    # nb_contour_list.append(nb_contour)

    # print(contour_length)

    if nb_contour == 1:
        result = 1
        print("C'est un pion")
    elif nb_contour == 2:
        result = 2
        print("C'est un cavalier")
    elif nb_contour == 5 and contour_length < 290:
        result = 3
        print("C'est un fou")
    elif (nb_contour == 5 or nb_contour == 6) and contour_length > 290:
        result = 4
        print("C'est une tour")
    elif nb_contour == 6:
        result = 6
        print("C'est un roi")
    else:
        result = 5
        print("C'est une dame")

    return result


def Caracteristic_black_contour(file):
    percent = 100
    piece_color = took_picture.piece_color(file)

    img2 = cv2.imread(file, cv2.IMREAD_COLOR)  # Lecture de l'image en couleurs
    img2 = resize_picture(img2, percent)

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Reading same image in another variable and converting to gray scale.
    img = resize_picture(img, percent)

    _, threshold = cv2.threshold(img, 0, 50, cv2.THRESH_BINARY)  # Converting image to a binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Detecting contours in image.

    xc = list()
    yc = list()
    contour_length = -260  # a peu près ca pourrait aussi être 1306 ... a cause de la taille de l'image de départ
    nb_contour = -1  # Car grand contour

    for cnt in contours:  # Going through every contours found in the image.
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        if cv2.arcLength(cnt, True) > 10:
            contour_length += cv2.arcLength(cnt, True)
            nb_contour += 1
            # print(cv2.arcLength(cnt, True))

        cv2.drawContours(img2, [approx], 0, (0, 0, 255), 1)

        n = approx.ravel()  # Used to flatted the array containing the co-ordinates of the vertices.
        i = 0

        for j in n:
            if i % 2 == 0:
                xc.append(n[i])
                yc.append(n[i + 1])

            i += 1

    return [xc, yc, nb_contour, contour_length]


def contour_black_piece_detection(file):
    xc, yc, nb_contour, contour_length = Caracteristic_black_contour(file)
    print(nb_contour, contour_length)

    if nb_contour == 1:
        result = 7
        print("C'est un pion")
    elif ((nb_contour == 3 or nb_contour == 2) and contour_length < 260 and contour_length > 200) :
        result = 8
        print("C'est un cavalier")
    elif nb_contour == 4 and contour_length < 360:
        result = 9
        print("C'est un fou")
    elif ((nb_contour == 6 or nb_contour == 5) and contour_length > 350 and contour_length < 452):
        result = 10
        print("C'est une tour")
    elif nb_contour > 9 or (nb_contour == 9 and contour_length > 370 and contour_length < 400) or \
            (nb_contour == 7 and contour_length > 250 and contour_length < 350) :
        result = 12
        print("C'est un roi")
    else:
        result = 11
        print("C'est une dame")

    return result


def Caracteristic_contour_from_image(img):
    _, threshold = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)  # Converting image to a binary image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Detecting contours in image.

    xc = list()
    yc = list()
    contour_length = 0

    nb_contour = 0
    for cnt in contours:  # Going through every contours found in the image.
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        # print(cv2.arcLength(cnt, True))
        if cv2.arcLength(cnt, True) > 5:
            contour_length += cv2.arcLength(cnt, True)
            nb_contour += 1

        # cv2.drawContours(img2, [approx], 0, (0, 0, 255), 1)

        n = approx.ravel()  # Used to flatted the array containing the co-ordinates of the vertices.
        i = 0

        for j in n:
            if i % 2 == 0:
                xc.append(n[i])
                yc.append(n[i + 1])
            i += 1

    # cv2.imshow('image2', img2)  # Showing the final image.

    return [xc, yc, nb_contour, contour_length]


def contour_white_piece_detection_from_image(img):
    # contour_length_list = list()
    # nb_contour_list = list()

    xc, yc, nb_contour, contour_length = Caracteristic_contour_from_image(img)
    # contour_length_list.append(contour_length)
    # nb_contour_list.append(nb_contour)

    # print(contour_length)

    if nb_contour == 1:
        result = 0
        print("C'est un pion")
    elif nb_contour == 2:
        result = 1
        print("C'est un cavalier")
    elif nb_contour == 5 and contour_length < 290:
        result = 2
        print("C'est un fou")
    elif (nb_contour == 5 or nb_contour == 6) and contour_length > 290:
        result = 3
        print("C'est une tour")
    elif nb_contour == 6:
        result = 5
        print("C'est un roi")
    else:
        result = 4
        print("C'est une dame")

    return result
