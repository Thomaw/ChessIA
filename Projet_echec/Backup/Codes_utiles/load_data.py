import took_picture
import cv2 as cv
import glob
import Constantes_values

images = [cv.imread(file) for file in glob.glob('Pieces_Validees/*.png')]
u = 0

#-------------------- Load data --------------------#
for image in images:

    for i in range(0, 8):
        for j in range(0, 8):
            carre_image = took_picture.obtain_a_square(image, i, j)
            # R = 140 ou 222 ou 146
            # G = 162 ou 227 ou 177
            # B = 173 ou 230 ou 102

            (b, g, r) = carre_image[round(Constantes_values.largeur_carre/2), round(Constantes_values.largeur_carre/2)]

            if b != 173 and b != 230 and b != 135 and b != 102:
                cv.imwrite(r'Pieces_en_vrac\ ' + str(u) + '.png', carre_image)
                u += 1
