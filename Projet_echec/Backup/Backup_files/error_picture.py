import cv2 as cv
import numpy as np
import glob


def error_calculation(files, img):
    width = int(img.shape[1])
    height = int(img.shape[0])
    dim = (width, height)

    sigma_list = list()

    for image in files:
        img2 = cv.imread(image)
        img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
        residue = img - img2
        # print(np.min(residue))

        sigma = np.sum(residue)
        sigma_list.append(sigma)

    min_error = min(sigma_list)
    print('min_error : ' + str(min_error))

    for image in files:
        img2 = cv.imread(image)
        img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
        residue = img - img2
        # print(residue)

        if np.sum(residue) == min_error:
            cv.imshow("image base de donnees", img2)
            cv.imshow("image de base", img)
            cv.imshow("NTM", residue)
            break

    cv.waitKey(0)

    return min_error


carre_image = cv.imread('../Data_Pictures/ 4.png')
mini_error_list = list()

files_Pion = glob.glob('Pieces_en_vrac/Blanc/Pion/*.png')
min_error_Pion = error_calculation(files_Pion,carre_image)
mini_error_list.append(min_error_Pion)

files_Cavalier = glob.glob('Pieces_en_vrac/Blanc/Cavalier/*.png')
min_error_Cavalier = error_calculation(files_Cavalier, carre_image)
mini_error_list.append(min_error_Cavalier)

files_Fou = glob.glob('Pieces_en_vrac/Blanc/Fou/*.png')
min_error_Fou = error_calculation(files_Fou, carre_image)
mini_error_list.append(min_error_Fou)

files_Tour = glob.glob('Pieces_en_vrac/Blanc/Tour/*.png')
min_error_Tour = error_calculation(files_Tour, carre_image)
mini_error_list.append(min_error_Tour)

files_Dame = glob.glob('Pieces_en_vrac/Blanc/Dame/*.png')
min_error_Dame = error_calculation(files_Dame, carre_image)
mini_error_list.append(min_error_Dame)

files_Roi = glob.glob('Pieces_en_vrac/Blanc/Roi/*.png')
min_error_Roi = error_calculation(files_Roi, carre_image)
mini_error_list.append(min_error_Roi)


print(mini_error_list)
