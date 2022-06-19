import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt

quantidade_de_reles = 3

#correntes de curto
I_curto = np.array([5000,4000,3000])

#funcao para calcular a constante de cada relé no caso linear
def constante (Icurto, Ipk, RTC):
    return 0.14/((Icurto/(RTC*Ipk))**0.02-1)

#iniciando vetor da funcao objetivo
c = np.zeros(quantidade_de_reles)

#calculando vetor da funcao objetivo
for i in range(quantidade_de_reles):
    c[i] = constante(I_curto[i],1,1000)


A = np.array([
    [      0,     -6.3019,  6.3019],
    [-6.3019,      6.3019,       0],
    [-4.9790,      4.9790,       0],
    [-4.2790,           0,       0],
    [      0,     -4.9790,       0],
    [      0,           0, -6.3019],
])

b = [-0.3, -0.3, -0.3, -0.2, -0.2, -0.2]

limites = [(0,1),(0,1),(0,1)]

#solução do problema linear
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
plt.loglog(correntes, tempos[:,2], "k.-")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha")
plt.legend(['R1','R2','R3'], loc="upper right")
plt.xlabel("Múltiplo")
plt.ylabel("Tempo [s]")
plt.grid(True, which="both", ls="-")
plt.show()
