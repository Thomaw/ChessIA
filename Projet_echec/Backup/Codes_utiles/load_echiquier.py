import took_picture
import cv2 as cv
import time

u = 125
while 1:
    took_picture.took_picture()
    crop_img = took_picture.read_image()

    cv.imwrite('Pieces_Validees\ ' + str(u) + '.png', crop_img)

    time.sleep(3)
    u += 1
