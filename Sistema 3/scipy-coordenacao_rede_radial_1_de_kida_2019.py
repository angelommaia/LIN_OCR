import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

quantidade_de_reles = 5

A = np.array([
    [-4.010, 3.754, 0, 0, 0],
    [-2.368, 3.029, 0, 0, 0],
    [-8.791, 0, 3.465, 0, 0],
    [-2.367, 0, 2.302, 0, 0],
    [0, -7.994, 0, 6.347, 0],
    [0, -3.476, 0, 1.597, 0],
    [0, 0, -6.124, 0, 5.157],
    [0, 0, -3.226, 0, 0.669],
    [-2.810, 0, 0, 0, 0],
    [-1.438, 0, 0, 0, 0],
    [0, -3.754, 0, 0, 0],
    [0, -3.029, 0, 0, 0],
    [0, 0, -3.465, 0, 0],
    [0, 0, -2.301, 0, 0],
    [0, 0, 0, -6.347, 0],
    [0, 0, 0, -1.597, 0],
    [0, 0, 0, 0, -5.157],
    [0, 0, 0, 0, -0.669],
    [2.810, 0, 0, 0, 0],
    [1.438, 0, 0, 0, 0],
    [0, 3.754, 0, 0, 0],
    [0, 3.029, 0, 0, 0],
    [0, 0, 3.465, 0, 0],
    [0, 0, 2.301, 0, 0],
    [0, 0, 0, 6.347, 0],
    [0, 0, 0, 1.597, 0],
    [0, 0, 0, 0, 5.156],
    [0, 0, 0, 0, 0.669]
    ])

b = [-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,2,2,2,2,2,2,2,2,2,2]

c = [4.289,6.783,5.767,7.944,5.826]

limites = [(0.1,10),(0.1,10),(0.1,10),(0.1,10),(0.1,10)]

res = linprog(c, A_ub=A, b_ub=b, bounds = limites)

print('Solução:', round(res.fun, ndigits=2),
      '\nvalores de x:', res.x,
      '\nNúmero de iterações:', res.nit,
      '\nStatus:', res.message)

#======PLOTAGEM DOS GRAFICOS======#

#gerando intervalo de corrente
correntes = np.linspace(900, 3000, num=43)

#intervalo de tempo
tempos = np.zeros((correntes.size,quantidade_de_reles))

#calculando o tempo de operacao para cada rele de acordo com o TDS calculado (pos[i])
for i in range(quantidade_de_reles):
    tempos[:,i] = res.x[i]*0.14/((correntes/(1000))**0.02-1)


plt.loglog(correntes, tempos[:,0],"k-")
plt.loglog(correntes, tempos[:,1], "k--")
plt.loglog(correntes, tempos[:,2], "k-.")

plt.title("Coordenograma dos Relés de Sobrecorrente das barras 1 e 2")
plt.legend(['R1', 'R2', 'R3'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])

plt.grid(True, which="both", ls="-")
plt.show()

plt.loglog(correntes, tempos[:,0], "k-")
plt.loglog(correntes, tempos[:,1],"k--")
plt.loglog(correntes, tempos[:,3], "k-.")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha 2")
plt.legend(['R1', 'R2', 'R4'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])

plt.grid(True, which="both", ls="-")
plt.show()

plt.loglog(correntes, tempos[:,0], "k-")
plt.loglog(correntes, tempos[:,2], "k--")
plt.loglog(correntes, tempos[:,4], "k-.")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha 1")
plt.legend(['R1', 'R3', 'R5'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])

plt.grid(True, which="both", ls="-")
plt.show()
