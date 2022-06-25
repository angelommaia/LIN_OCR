import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters import (plot_cost_history,
                                    plot_contour, plot_surface)
np.random.seed(15)
#constantes do problema
quantidade_de_reles = 3
CTI=0.3

#constantes do método
n_particles = 200
iteracoes = 2500
quantidade_de_sementes = 1
peso = 1

def funcao_objetivo(x):
    #funcao objetivo
    S = 4.279*x[:, 0] + 4.979*x[:,1] + 6.3019*x[:,2]

    #penalidades de coordenacao
    p1 = np.zeros(n_particles)
    p2 = np.zeros(n_particles)
    p3 = np.zeros(n_particles)

    for particula in range(n_particles):
        #restricoes
        #6.3019x2 – 6.3019x3 >= 0.3 -> 0.3 + 6.3019x3 - 6.3019x2 <= 0
        if (6.3019*x[particula,1] - 6.3019*x[particula,2] < 0.3):
            p1[particula] = p1[particula] + 1
        #6.3019x1 – 6.3019x2 >= 0.3 -> 0.3 + 6.3019x2 - 6.3019x1 <= 0
        if (6.3019*x[particula,0] - 6.3019*x[particula,1] < 0.3):
            p2[particula] = p2[particula] + 1
        #4.9790x1 – 4.9790x2 >= 0.3 -> 0.3 +  4.9790x2 - 4.9790x1 < = 0
        if (4.9790*x[particula,0] - 4.9790*x[particula,1] < 0.3):
            p2[particula] = p2[particula] + 1

    return S + (p1+p2+p3)*peso

#limites para as variaveis
x_max = 1 * np.ones(3)
x_min = [0.2/4.2790, 0.2/4.2790, 0.2/6.3019]
bounds = (x_min, x_max)

options = {'c1': 2, 'c2': 2, 'w':0.3}
optimizer = GlobalBestPSO(n_particles, dimensions=quantidade_de_reles, options=options, bounds=bounds)


now = datetime.now()
historico = open('historico_otimizacao_{}.txt'.format(now.strftime("%d%m%Y%H%M%S")), 'a')
historico.write("Inicio em: {}\n".format(now.strftime("%H:%M:%S")))

for semente_otimizador in range(quantidade_de_sementes):
    #otimizador

    cost, pos = optimizer.optimize(funcao_objetivo, iteracoes)
    historico.write('{:3.4f} {}\n'.format(cost,pos))

now = datetime.now()
historico.write("Fim em: {}".format(now.strftime("%H:%M:%S")))
historico.close()

#======PLOTAGEM DOS GRAFICOS======#

#gerando intervalo de corrente
correntes = np.linspace(1000, 5000, num=43)

#intervalo de tempo
tempos = np.zeros((correntes.size,quantidade_de_reles))

#calculando o tempo de operacao para cada rele de acordo com o TDS calculado (pos[i])
for i in range(quantidade_de_reles):
    tempos[:,i] = pos[i]*0.14/((correntes/(1000))**0.02-1)


plt.loglog(correntes, tempos[:,0],"k-")
plt.loglog(correntes, tempos[:,1], "k--")
plt.loglog(correntes, tempos[:,2], "k-.")

plt.title("Coordenograma dos Relés de Sobrecorrente da Linha")
plt.legend(['R1','R2','R3'], loc="upper right")
plt.xlabel("Corrente [A]")
plt.ylabel("Tempo [s]")
plt.yticks(ticks=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200], labels=[0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])
plt.grid(True, which="both", ls="-")
plt.show()

my_designer = Designer(legend='',  label=['Número de Iterações','Custo da Função Objetivo'], text_fontsize='medium', title_fontsize='20', figsize=(8,6))
plot_cost_history(cost_history=optimizer.cost_history, title='Histórico do custo para o Sistema 1', designer = my_designer)
plt.show()
