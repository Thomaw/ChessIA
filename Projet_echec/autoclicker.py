import math
import pyautogui
import time
import took_picture


# https://pyautogui.readthedocs.io/en/latest/mouse.html
def Leftclick(x, y):
    """Fonction pour rempalcer le clic de la souris"""
    pyautogui.click(x=x, y=y)  # move to (x,y), then click the left mouse button.


def white_deplacement_Lichess(pa, pv):
    """Fonction pour déplacer une pièce pour les blancs"""
    xa = math.floor(pa / 8)
    ya = pa % 8

    xv = math.floor(pv / 8)
    yv = pv % 8

    id, jd = took_picture.position_carre(7 - ya, xa)
    Leftclick(id, jd)
    time.sleep(1)
    ia, ja = took_picture.position_carre(7 - yv, xv)
    Leftclick(ia, ja)


def black_deplacement_Lichess(pa, pv):
    """Fonction pour déplacer une pièce pour les noirs"""
    xa = math.floor(pa / 8)
    ya = pa % 8

    xv = math.floor(pv / 8)
    yv = pv % 8

    # print(xa, ya, xv, yv)

    id, jd = took_picture.position_carre(ya, 7 - xa)
    Leftclick(id, jd)
    time.sleep(1)
    ia, ja = took_picture.position_carre(yv, 7 - xv)
    Leftclick(ia, ja)

### FONCTION A VERIFIER car jamais utilisée
def resign_click():
    """Fonction pour quitter une partie"""

    # xl, yl est la position du bouton de resign

    # Pas sur que le bouton resign soit toujours au meme endroit en fonction du temps que l'on a

    xl, yl = 1620, 680
    Leftclick(xl, yl)
    Leftclick(xl, yl)

