import Board_indentification
import took_picture
import os
import chess

board = Board_indentification.board_pieces_detection()
Board_indentification.affichage_echiquier(board)

# On suppose que toutes nos parties commence avec un board logique et complet
Color_play = took_picture.piece_color('Data_Pictures/ 63.png')
print(Color_play)
"""
Pour Color_play : 
- True : On joue blanc
- False : On joue noir
"""


# On a réussi à modifier l'algo pour afficher le board de la bonne façon
print('play as : ', Color_play)
chess.play_as(Color_play)


# Fait de l'espace dans le disque
# os.remove(r'Data_Pictures\test.jpg')


