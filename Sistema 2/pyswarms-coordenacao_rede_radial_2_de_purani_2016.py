import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import (plot_cost_history,
                                    plot_contour, plot_surface)
#constantes do problema
quantidade_de_reles = 4
CTI=0.25

#constantes do método
numero_de_particulas = 350
iteracoes = 6000
quantidade_de_sementes = 1
peso = 10 #peso das penalidades

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

options = {'c1': 2, 'c2': 5, 'w':0.7}
optimizer = GlobalBestPSO(numero_de_particulas, dimensions=quantidade_de_reles, options=options, bounds=bounds)

#otimizador
for semente_otimizador in range(quantidade_de_sementes):
    np.random.seed(3)
    cost, pos = optimizer.optimize(funcao_objetivo, iteracoes)
