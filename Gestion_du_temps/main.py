import time
import Clock_function

alpha = 0.205       # Facteur d'atténuation
k = 0              # Nombre de coup réalisé

start_time = time.time()

while 1 :
    Time_management = Clock_function.Players_Time('AI_model\model_40.json', 'AI_model\model_40.h5')

    print('Temps adverse : ', Time_management[0]) # Toujours en premier l'image du haut, donc le temps de l'adversaire
    print('Notre temps : ', Time_management[1])
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    print("--------------------------------------------------------")

    theorical_time_calculation = Clock_function.theoretical_time(k, alpha)
    print('Theorical time :', theorical_time_calculation)

    time.sleep(0.1)
    k +=1
