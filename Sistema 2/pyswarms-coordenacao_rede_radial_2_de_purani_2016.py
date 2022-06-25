import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters import (plot_cost_history,
                                    plot_contour, plot_surface)
#constantes do problema
quantidade_de_reles = 4
CTI=0.25
np.random.seed(2)
#constantes do método
numero_de_particulas = 250
iteracoes = 4000
quantidade_de_sementes = 1
peso = 1 #peso das penalidades

def funcao_objetivo(x):
    #funcao objetivo
    S = 4.2797*x[:, 0] + 5.7492*x[:,1] + 4.2797*x[:,2] + 5.7492*x[:,3]

    #penalidades de coordenacao
    p1 = np.zeros(numero_de_particulas)
    p2 = np.zeros(numero_de_particulas)

    for particula in range(numero_de_particulas):

        #13.6280x3 – 5.7492x2 >= 0.25
        if (13.6280*x[particula,2] - 5.7492*x[particula,1] < 0.25):
            p1[particula] = p1[particula] + 1
        #13.6280x1 – 5.7492x4 >= 0.25
        if (13.6280*x[particula,0] - 5.7492*x[particula,3] < 0.25):
            p1[particula] = p1[particula] + 1

    return S + (p1+p2)*peso

#limites para as variaveis
x_max = 1 * np.ones(quantidade_de_reles)
x_min = [0.2/4.2797, 0.2/5.7492, 0.2/4.2797, 0.2/5.7492]
bounds = (x_min, x_max)

options = {'c1': 2, 'c2': 3, 'w':0.8}
optimizer = GlobalBestPSO(numero_de_particulas, dimensions=quantidade_de_reles, options=options, bounds=bounds)

#otimizador
for semente_otimizador in range(quantidade_de_sementes):

    cost, pos = optimizer.optimize(funcao_objetivo, iteracoes)
    my_designer = Designer(legend='',  label=['Número de Iterações','Custo da Função Objetivo'], text_fontsize='medium', title_fontsize='20', figsize=(8,6))
    plot_cost_history(cost_history=optimizer.cost_history, title='Histórico do custo para o Sistema 2', designer = my_designer)
    plt.show()

#======PLOTAGEM DOS GRAFICOS======#

#gerando espaço de correntes de 4000 ate 800 [A] para utilizar na plotagem
correntes = np.linspace(1000, 5000, num=43)

tempos = np.zeros((correntes.size,quantidade_de_reles))

#melhor tempo
x= [0.0497, 0.0349, 0.0472, 0.0363]

for i in range(quantidade_de_reles):
    tempos[:,i] = x[i]*0.14/((correntes/(1000))**0.02-1)


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
