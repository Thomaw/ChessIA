import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import took_picture


# images = [cv2.imread(file) for file in glob.glob('Pieces_en_vrac/*.png')]
images = [cv2.imread(file) for file in glob.glob('Pieces_noires/*.png')]
# images = [cv2.imread(file) for file in glob.glob('Pieces_blanches/*.png')]
# images = [cv2.imread(file) for file in glob.glob('Pieces_par_couleur_et_forme/*.png')]


White_lower_bound = np.array([252, 252, 252])
White_upper_bound = np.array([255, 255, 255])

Black_lower_bound = np.array([0, 0, 0])
Black_upper_bound = np.array([3, 3, 3])

x = list()
y = list()

for image in images:
    White_msk = cv2.inRange(image, White_lower_bound, White_upper_bound)
    Black_msk = cv2.inRange(image, Black_lower_bound, Black_upper_bound)

    percent_white = took_picture.calcPercentage(White_msk)
    percent_black = took_picture.calcPercentage(Black_msk)

    x.append(percent_white)
    y.append(percent_black)


plt.scatter(x, y)

plt.title('Nuage de points avec Matplotlib')
plt.xlabel('Pourcentage de blanc sur l''image')
plt.ylabel('Pourcentage de noir sur l''image')
plt.show()

