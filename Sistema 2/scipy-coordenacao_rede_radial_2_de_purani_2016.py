import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

quantidade_de_reles = 4

A = np.array([
[       0, 5.7492, -13.6280,      0],
[-13.6280,      0,        0, 5.7492],
[ -4.2797,      0,        0,      0],
[       0,-5.7492,        0,      0],
[       0,      0,  -4.2797,      0],
[       0,      0,        0,-5.7492]
])

b = [-0.25, -0.25, -0.2, -0.2, -0.2, -0.2]

c = [4.2797,5.7492,4.2797,5.7492]

limites = [(0,1),(0,1),(0,1),(0,1)]

#solucão do problema linear
res = linprog(c, A_ub=A, b_ub=b, bounds = limites, method="simplex")


print('Solução:', round(res.fun, ndigits=2),
      '\nvalores de x:', res.x,
      '\nNúmero de iterações:', res.nit,
      '\nStatus:', res.message)

#======PLOTAGEM DOS GRAFICOS======#

#gerando espaço de correntes de 4000 ate 800 [A] para utilizar na plotagem
correntes = np.linspace(1000, 5000, num=43)

tempos = np.zeros((correntes.size,quantidade_de_reles))

for i in range(quantidade_de_reles):
    tempos[:,i] = res.x[i]*0.14/((correntes/(1000))**0.02-1)


plt.loglog(correntes, tempos[:,0],"k-")
plt.loglog(correntes, tempos[:,3], "k--")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha 1")
plt.legend(['R1','R4'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])
plt.grid(True, which="both", ls="-")
plt.show()

plt.loglog(correntes, tempos[:,2],"k-")
plt.loglog(correntes, tempos[:,1], "k--")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha 2")
plt.legend(['R3','R2'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])
plt.grid(True, which="both", ls="-")
plt.show()
