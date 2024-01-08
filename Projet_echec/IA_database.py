import glob
import cv2 as cv


"""
Etape de création de l'IA:
- Recréer une base de données avec les pièces           : Fait mais pas sur
- Toutes les passer le thresholding en récupérant th3   : Fait mais pas sur
- Créer l'iA
- L'entrainer avec les pièces de la base de données
"""


def thresholding(image):
    img = cv.medianBlur(image, 5)

    # ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # images = [img, th1, th2, th3]
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    # return img, th1, th2, th3
    return th3


u = 1
for f in glob.glob('Pieces_en_vrac/Reference/Tour/*'):
    image = cv.imread(f, 0)
    # img, th1, th2, th3 = thresholding(image)

    th3 = thresholding(image)
    cv.imwrite(r'Pieces_en_vrac/Threshold/Tour/ ' + str(u) + '.png', th3)
    # cv.imshow('img', th3)
    # cv.waitKey(0)
    u += 1
