import numpy as np #vetores, matrizes, aleatoriedade
import matplotlib.pyplot as plt #graficos
import sys
from datetime import datetime #aquisição de hora e data para estampa de tempo
import pandas as pd #apresentação dos dados em forma de tabela
from tabulate import tabulate #apresentação dos dados em forma de tabela
from IPython.display import Image

#otimização
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import (plot_cost_history,
                                    plot_contour, plot_surface)

#formatacao dos prints de arrays numpy
np.set_printoptions(suppress=True, formatter={'\
                                                float_kind': '{:f}'.format})


#==============================================================================#
#===========================Função Tempo De Operação===========================#
#==============================================================================#


#função para o tempo de operacao de um rele de sobrecorrente (top)
def top(TDS_i, I_pick_up_i, corrente_no_sistema, RTC):
    return TDS_i*0.14/((corrente_no_sistema/(I_pick_up_i*RTC))**0.02-1)


#==============================================================================#
#====================================Global====================================#
#============================Parâmetros do Circuito============================#
#==============================================================================#

#numero total de relés usados na topologia
quantidade_de_reles = 8

#Relações entre relés de sobrecorrente
Rele_principal = np.array([1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                           5, 5, 6, 6, 7, 7, 8, 8])

Rele_secundario = np.array([1, 2, 5, 7, 3, 7, 1, 4, 5, 1,
                            5, 8, 6, 3, 7, 6, 8, 4])

#Correntes de curto circuito
'''
Cada elemento dos vetores apresenta a corrente vista pelo relé que está no vetor
Rele_secundario para um curto no relé de índice correspondente contido no vetor
Rele_principal.
'''
#Curto Close-in
I_curto_close_in = np.array([6476.2, 2705.2, 1200.2, 1505.1, 4879.2, 1505.1,
                             3381.2, 4576.6, 1200.2, 3381.2, 9648.3, 1008.4,
                             9451.2, 804.1, 3502.9, 3502.9, 1764.2, 1764.2])

I_curto_80 = np.array([3742.1, 2061.7, 914.7, 1147.1, 1261.0, 165.3,
                       1415.0, 2049.3, 285.3, 1764.0, 1575.0, 822.7,
                       4023.9, 35.9, 1751.6, 1751.6, 1192.9, 1192.9])

I_curto_barra_remota = np.array([3381.2, 1946.6, 863.6, 1083.0, 807.6, 1000.0,
                                 1807.5, 1765.5, 125.7, 1641.4, 1202.0, 1499.8,
                                 3505.2, 124.3, 1506.3, 1506.3, 1009.5, 1009.5])

##vetor com menores e maiores correntes de curto
I_menores_correntes = np.zeros(I_curto_close_in.size)
I_maiores_correntes = np.zeros(I_curto_close_in.size)

for i in range(I_curto_close_in.size):
    aux = np.array([I_curto_close_in[i],I_curto_80[i],I_curto_barra_remota[i]])
    I_menores_correntes[i] = min(aux)
    I_maiores_correntes[i] = max(aux)


#Relações de transformação dos TCs
RTC = np.array([1, 1, 1, 1, 1, 1, 1, 1])

#intervalos de coordenação
CTI1 = 0.3
CTI2 = 0.2

#==============================================================================#
#====================================Global====================================#
#=======================Limites das Variáveis de Decisão=======================#
#==============================================================================#
#TDS
min_bound_TDS = 0.01*np.ones((quantidade_de_reles))  # limites min e max dados
max_bound_TDS = 1.0*np.ones((quantidade_de_reles))

#Ipk
min_bound_Ip = 600*np.ones((quantidade_de_reles))  # Inominal*1.2
max_bound_Ip = 800*np.ones((quantidade_de_reles))

#TRD
min_bound_TRD = 0.3*np.ones((quantidade_de_reles))
max_bound_TRD = 0.8*np.ones((quantidade_de_reles))

'''
Os vetores de limite devem ser concatenados e posteriormente passados para uma
matriz que será passada como argumento da função objetivo.
'''
min_bound = np.append(np.append(min_bound_TDS, min_bound_Ip), min_bound_TRD)
max_bound = np.append(np.append(max_bound_TDS, max_bound_Ip), max_bound_TRD)

#matriz de limites maximos e minimos para todas as variaveis
bounds = (min_bound, max_bound)

#==============================================================================#
#====================================Global====================================#
#===========================Parâmetros da Otimização===========================#
#==============================================================================#

options = {'c1': 2, 'c2': 2, 'w':0.4}

numero_de_particulas = 15
quantidade_de_sementes = 50
quantidade_de_iteracoes = 5000

#==============================================================================#
#====================================Função====================================#
#================================Função Objetivo===============================#
#==============================================================================#
def funcao_objetivo(x):

    '''
    O algoritmo da biblioteca passa uma matriz contendo em cada coluna todas as
    partículas atuais para uma determinada variável de decisão, ou seja;
    x[<numero de partículas>,<variável>]. O primeiro for desse código passa cada
    coluna dessa matriz de entrada para uma variável correspondente, para
    facilitar a visualização do que está acontecendo.

    As variáveis são empilhadas, ou postas lado a lado, com o uso da função
    np.column_stack. Assim, cada coluna "i" da nova matriz TDS representa o
    valor de TDS no relé "i+1" para cada partícula da iteração atual.

    '''
#==============================================================================#
#=========================Atribuição Inicial De Valores========================#
#==============================================================================#

    #primeira coluna fora do laço para que o np.column_stack funcione
    TDS = x[:,0]
    Ipk = x[:,0+quantidade_de_reles]
    TRD = x[:,0+quantidade_de_reles*2]

    #todas as particulas de cada variavel de decisão como uma coluna nas matrizes
    for i in range(quantidade_de_reles-1):
        TDS = np.column_stack((TDS, x[:,i+1]))
        Ipk = np.column_stack((Ipk, x[:,i+1+quantidade_de_reles]))
        TRD = np.column_stack((TRD, x[:,i+1+quantidade_de_reles*2]))

    '''
        pegar o maior valor de corrrente de curto e substituir o Ipk por ela quando
        Ipk for menor do que ela

        colocar a restrição para tempo negativo

    '''
    '''
    for particula in range(numero_de_particulas):
        for i in range(quantidade_de_reles):
            if Ipk[particula,i] < I_maiores_correntes[i]:
                Ipk[particula,i] =  I_maiores_correntes[i]+1

    '''

    for particula in range(numero_de_particulas):
        for i in range(quantidade_de_reles):
            if Ipk[particula,i] > I_menores_correntes[i]:
                Ipk[particula,i] =  I_menores_correntes[i]+100



#==============================================================================#
#=================Tempos de Operacao dos Relés de Sobrecorrente================#
#===========================Para os Curtos Informados==========================#
#==============================================================================#

#==============================================================================#

    #tempos de curto close in
    Ilida = I_curto_close_in

    tempo_operacao_close_in = TDS[:,0]*0.14/((Ilida[0]/(Ipk[:,0]*RTC[0]))**0.02-1)

    for i in range(quantidade_de_reles-1):

        tempo_aux = TDS[:,i+1]*0.14/((Ilida[i+1]/(Ipk[:,i+1]*RTC[i+1]))**0.02-1)
        tempo_operacao_close_in = np.column_stack((tempo_operacao_close_in,
                                                   tempo_aux))

#==============================================================================#

    #tempos de curto 80%
    Ilida = I_curto_80

    tempo_operacao_80 = TDS[:,0]*0.14/((Ilida[0]/(Ipk[:,0]*RTC[0]))**0.02-1)

    for i in range(quantidade_de_reles-1):

        tempo_aux = TDS[:,i+1]*0.14/((Ilida[i+1]/(Ipk[:,i+1]*RTC[i+1]))**0.02-1)
        tempo_operacao_80 = np.column_stack((tempo_operacao_80,
                                                   tempo_aux))

#==============================================================================#

    #tempos de curto barra remota
    Ilida = I_curto_barra_remota

    tempo_operacao_barra_remota =\
     TDS[:,0]*0.14/((Ilida[0]/(Ipk[:,0]*RTC[0]))**0.02-1)

    for i in range(quantidade_de_reles-1):

        tempo_aux = TDS[:,i+1]*0.14/((Ilida[i+1]/(Ipk[:,i+1]*RTC[i+1]))**0.02-1)
        tempo_operacao_barra_remota = np.column_stack(
                                        (tempo_operacao_barra_remota,tempo_aux))

#==============================================================================#
#=======================Somatório dos Tempos de Operação=======================#
#==============================================================================#

    #Somatório dos tempos de Operação Close In para a Função Objetivo
    somatorio_tempo_de_operacao_close_in = np.zeros(numero_de_particulas)
    #for particula in range(numero_de_particulas):
    for i in range(quantidade_de_reles):
        somatorio_tempo_de_operacao_close_in +=  tempo_operacao_close_in[:,i]

    #Somatório dos tempos de operação dos relés de distância
    somatorio_tempo_de_operacao_rele_distancia = np.zeros(numero_de_particulas)

    for i in range(quantidade_de_reles):
        somatorio_tempo_de_operacao_rele_distancia += TRD[:,i]


    func_obj = somatorio_tempo_de_operacao_close_in +\
                         somatorio_tempo_de_operacao_rele_distancia

#==============================================================================#
#=================================Penalidades==================================#
#==============================================================================#

    '''
    As penalidades são uma maneira mais branda de modelar restrições uma vez que
    o otimizador ainda pode visitar os valores não desejados. Nesse caso, ocorre
    a adição de uma penalidade no somatório da função objetivo para cada vez que
    uma restrição modelada como penalidade é desrespeitada.
    '''
#==============================================================================#
#==========================Penalidade CTI Close In=============================#
#==============================================================================#

    #Penalidade para o tempo de coordenação Close In
    penalidade_CTI_close_in=np.zeros(numero_de_particulas)
    Ilida = I_curto_close_in

    #Laço mais externo inidca que o processo iterativo deve ocorrer p/ o número
    #de relações principal-backup existente no sistema
    for i in range(Rele_principal.size):
        #O for interno percorre as n particulas da iteração
        for particula in range(numero_de_particulas):

            #variáveis auxiliares para despoluir o código
            RP = Rele_principal[i]
            RB = Rele_secundario[i]

            if(RP!=RB):

                tempo_principal = top(TDS[:,RP-1],Ipk[:,RP-1],
                                                        Ilida[RP-1],RTC[RP-1])
                tempo_bkp = top(TDS[:,RB-1],Ipk[:,RB-1],Ilida[RB-1],RTC[RB-1])

        #a penalidade para o tempo de coordenação close in consiste na diferença
        #entre o CTI mínimo e o tempo de coordenação atual. Caso o tempo
        #atual seja menor do que o mínimo. Caso seja maior, a penalidade é nula.
                if tempo_bkp[particula] - tempo_principal[particula] < 0.3:
                    penalidade_CTI_close_in[particula]=\
                                penalidade_CTI_close_in[particula] + 1

#==============================================================================#
#============================Penalidade CTI 80%================================#
#==============================================================================#

    #Penalidade para o tempo de coordenação 80%
    penalidade_CTI_80=np.zeros(numero_de_particulas)
    Ilida = I_curto_80

#Laço mais externo inidca que o processo iterativo deve ocorrer para o número
#de relações principal-backup existente no sistema
    for i in range(Rele_principal.size):
        #O for interno percorre as n particulas da iteração
        for particula in range(numero_de_particulas):

            #variáveis auxiliares para despoluir o código
            RP = Rele_principal[i]
            RB = Rele_secundario[i]

            if(RP!=RB):

                tempo_principal = top(TDS[:,RP-1],Ipk[:,RP-1],Ilida[RP-1],
                                                                    RTC[RP-1])
                tempo_bkp = top(TDS[:,RB-1],Ipk[:,RB-1],Ilida[RB-1],RTC[RB-1])

        #a penalidade para o tempo de coordenação close in consiste na diferença
        #entre o CTI mínimo e o tempo de coordenação atual. Caso o tempo
        #atual seja menor do que o mínimo. Caso seja maior, a penalidade é nula.
                if tempo_bkp[particula] - tempo_principal[particula] < 0.3:
                    penalidade_CTI_80[particula]=\
                                penalidade_CTI_80[particula] + 1
#==============================================================================#
#========================Penalidade CTI Barra Remota===========================#
#==============================================================================#

    #Penalidade para o tempo de coordenação 80%
    penalidade_CTI_barra_remota=np.zeros(numero_de_particulas)
    Ilida = I_curto_barra_remota

    #Laço mais externo inidca que o processo iterativo deve ocorrer p/ o número
    #de relações principal-backup existente no sistema
    for i in range(Rele_principal.size):
        #O for interno percorre as n particulas da iteração
        for particula in range(numero_de_particulas):

            #variáveis auxiliares para despoluir o código
            RP = Rele_principal[i]
            RB = Rele_secundario[i]

            if(RP!=RB):

                tempo_principal = top(TDS[:,RP-1],Ipk[:,RP-1],Ilida[RP-1],
                                                                    RTC[RP-1])
                tempo_bkp = top(TDS[:,RB-1],Ipk[:,RB-1],Ilida[RB-1],RTC[RB-1])

        #a penalidade para o tempo de coordenação close in consiste na diferença
        #entre o CTI mínimo e o tempo de coordenação atual. Caso o tempo
        #atual seja menor do que o mínimo. Caso seja maior, a penalidade é nula.
                if tempo_bkp[particula] - tempo_principal[particula] < 0.3:
                    penalidade_CTI_barra_remota[particula]=\
                                penalidade_CTI_barra_remota[particula] + 1

#==============================================================================#
#==========================Penalidade RDS Close In=============================#
#==============================================================================#

    #Penalidade para o tempo de coordenação entre relé de sobrecorrente e relé
    #de distância para curtos close in
    penalidade_CTI2_close_in=np.zeros(numero_de_particulas)
    Ilida = I_curto_close_in

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        for particula in range(numero_de_particulas):

            RDIR = Rele_principal[i]
            RDIST = TRD[particula,i]

            tempo_direcional = top(TDS[:,RDIR-1],Ipk[:,RDIR-1],Ilida[RDIR-1],
                                                                RTC[RDIR-1])

            if tempo_direcional[particula] - RDIST < CTI2:
                penalidade_CTI2_close_in[particula]=\
                            penalidade_CTI2_close_in[particula] + 1
#==============================================================================#
#=============================Penalidade RDS 80%===============================#
#==============================================================================#

    #Penalidade para o tempo de coordenação entre relé de sobrecorrente e relé
    #de distância para curtos close in
    penalidade_CTI2_80=np.zeros(numero_de_particulas)
    Ilida = I_curto_80

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        for particula in range(numero_de_particulas):

            RDIR = Rele_principal[i]
            RDIST = TRD[particula,i]

            tempo_direcional = top(TDS[:,RDIR-1],Ipk[:,RDIR-1],Ilida[RDIR-1],
                                                                RTC[RDIR-1])

            if tempo_direcional[particula] - RDIST < CTI2:
                penalidade_CTI2_80[particula]=\
                            penalidade_CTI2_80[particula] + 1
#==============================================================================#
#========================Penalidade RDS Barra Remota===========================#
#==============================================================================#

    #Penalidade para o tempo de coordenação entre relé de sobrecorrente e relé
    #de distância para curtos close in
    penalidade_CTI2_barra_remota=np.zeros(numero_de_particulas)
    Ilida = I_curto_barra_remota

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        for particula in range(numero_de_particulas):

            RDIR = Rele_principal[i]
            RDIST = TRD[particula,i]

            tempo_direcional = top(TDS[:,RDIR-1],Ipk[:,RDIR-1],Ilida[RDIR-1],
                                                                RTC[RDIR-1])
            if tempo_direcional[particula] - RDIST < CTI2:
                penalidade_CTI2_barra_remota[particula]=\
                            penalidade_CTI2_barra_remota[particula] + 1

#==============================================================================#
    penalidades = penalidade_CTI_close_in*peso_penalidade[0] +\
                  penalidade_CTI_80*peso_penalidade[1] +\
                  penalidade_CTI_barra_remota*peso_penalidade[2] +\
                  penalidade_CTI2_close_in*peso_penalidade[3] +\
                  penalidade_CTI2_80*peso_penalidade[4] +\
                  penalidade_CTI2_barra_remota*peso_penalidade[5]

    return func_obj + penalidades

'''
#==============================================================================#
#====================================Verbose===================================#
#================================Função Objetivo===============================#
#==============================================================================#
'''
def funcao_objetivo_verbose(x):

    #inicialização de variáveis locais
    soma_tempo_de_operacao_close_in = 0
    soma_tempo_de_operacao_80  = 0
    TDS = np.zeros(quantidade_de_reles)
    Ipk = np.zeros(quantidade_de_reles)
    TRD = np.zeros(quantidade_de_reles)

    #atribuição das entradas em vetores correspondentes às variáveis de decisão
    for i in range(quantidade_de_reles):
        TDS[i] = x[i]
        Ipk[i] = x[i+quantidade_de_reles]
        TRD[i] = x[i+2*quantidade_de_reles]

#==========================Somatório Função Objetivo===========================#

    #soma_tempo_de_operacao_close_in
    Ilida = I_curto_close_in

    for i in range(quantidade_de_reles):
        soma_tempo_de_operacao_close_in += TDS[i]*0.14/((
                Ilida[i]/(Ipk[i]*RTC[i]))**0.02-1)
        soma_tempo_de_operacao_80 += TRD[i]

    func_obj = soma_tempo_de_operacao_close_in + soma_tempo_de_operacao_80

#=================================Penalidades==================================#

    print("#======Penalidades======#")
    #close in
    Ilida = I_curto_close_in
    penalidade_CTI_close_in = 0

    for i in range(Rele_principal.size):
        RP = Rele_principal[i]
        RB = Rele_secundario[i]
        if(RP!=RB):
            tempo_principal = top(TDS[RP-1], Ipk[RP-1], Ilida[RP-1],RTC[RP-1])
            tempo_bkp = top(TDS[RB-1], Ipk[RB-1], Ilida[RB-1],RTC[RB-1])
            penalidade_CTI_close_in = penalidade_CTI_close_in +\
                                      max(0, CTI1 + tempo_principal - tempo_bkp)
            if (max(0, CTI1 + tempo_principal - tempo_bkp)):
                    print("close-in relés de sobre corrente")
                    print("relé principal ",Rele_principal[i]," tempo=",
                                                                tempo_principal)
                    print("relé backup ",Rele_secundario[i]," tempo=",tempo_bkp)
                    print("Penalidade Aplicada: ",max(0,
                         CTI1 + tempo_principal - tempo_bkp)*peso_penalidade[0])
                    print("")
    #80%
    Ilida = I_curto_80
    penalidade_CTI_80 = 0

    for i in range(Rele_principal.size):
        RP = Rele_principal[i]
        RB = Rele_secundario[i]
        if(RP!=RB):
            tempo_principal = top(TDS[RP-1], Ipk[RP-1], Ilida[RP-1],RTC[RP-1])
            tempo_bkp = top(TDS[RB-1], Ipk[RB-1], Ilida[RB-1],RTC[RB-1])
            penalidade_CTI_80 = penalidade_CTI_80 +\
                                      max(0, CTI1 + tempo_principal - tempo_bkp)
            if (max(0, CTI1 + tempo_principal - tempo_bkp)):
                    print("80% remota relés de sobre corrente")
                    print("relé principal ",Rele_principal[i]," tempo=",
                                                                tempo_principal)
                    print("relé backup ",Rele_secundario[i]," tempo=",tempo_bkp)
                    print("Penalidade Aplicada: ",max(0,
                         CTI1 + tempo_principal - tempo_bkp)*peso_penalidade[1])
                    print("")

    #barra remota
    Ilida = I_curto_barra_remota
    penalidade_CTI_barra_remota = 0

    for i in range(Rele_principal.size):
        RP = Rele_principal[i]
        RB = Rele_secundario[i]
        if(RP!=RB):
            tempo_principal = top(TDS[RP-1], Ipk[RP-1], Ilida[RP-1],RTC[RP-1])
            tempo_bkp = top(TDS[RB-1], Ipk[RB-1], Ilida[RB-1],RTC[RB-1])
            penalidade_CTI_barra_remota = penalidade_CTI_barra_remota +\
                                                                max(0, CTI1 + tempo_principal - tempo_bkp)
            if (max(0, CTI1 + tempo_principal - tempo_bkp)):
                    print("Barra remota remota relés de sobre corrente")
                    print("relé principal ",Rele_principal[i]," tempo=",tempo_principal)
                    print("relé backup ",Rele_secundario[i]," tempo=",tempo_bkp)
                    print("Penalidade Aplicada: ",max(0, CTI1 + tempo_principal - tempo_bkp)*peso_penalidade[2])
                    print("")

#==============================================================================#
#==========================Penalidade RDS Close In=============================#
#==============================================================================#
    #Penalidade para o tempo de coordenação entre relé de sobrecorrente e relé
    #de distância para curtos close in
    penalidade_CTI2_close_in = 0
    Ilida = I_curto_barra_remota

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        RDIR = Rele_principal[i]
        RDIST = TRD[i]

        tempo_direcional = top(TDS[RDIR-1],Ipk[RDIR-1],Ilida[RDIR-1],
                                                            RTC[RDIR-1])
        penalidade_CTI2_close_in = \
                                penalidade_CTI2_close_in +\
                                max(0, CTI2 + RDIST -\
                                tempo_direcional)
#==============================================================================#
#=============================Penalidade RDS 80%===============================#
#==============================================================================#

    #coordenacao distancia 80%
    penalidade_CTI2_80 = 0
    Ilida = I_curto_80

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        RDIR = Rele_principal[i]
        RDIST = TRD[i]

        tempo_direcional = top(TDS[RDIR-1],Ipk[RDIR-1],Ilida[RDIR-1],
                                                            RTC[RDIR-1])
        penalidade_CTI2_80 = \
                                penalidade_CTI2_80 +\
                                max(0, CTI2 + RDIST -\
                                tempo_direcional)
#==============================================================================#
#=============================Penalidade Barra Remota==========================#
#==============================================================================#

    #coordenacao distancia 80%
    penalidade_CTI2_barra_remota = 0
    Ilida = I_curto_barra_remota

    #esse para cara rele de distancia existe um rele de sobrecorrente
    for i in range(quantidade_de_reles-1):

        RDIR = Rele_principal[i]
        RDIST = TRD[i]

        tempo_direcional = top(TDS[RDIR-1],Ipk[RDIR-1],Ilida[RDIR-1],
                                                            RTC[RDIR-1])
        penalidade_CTI2_barra_remota = \
                                penalidade_CTI2_barra_remota +\
                                max(0, CTI2 + RDIST -\
                                tempo_direcional)
#==============================================================================#

    penalidades = penalidade_CTI_close_in*peso_penalidade[0] +\
                  penalidade_CTI_80*peso_penalidade[1] +\
                  penalidade_CTI_barra_remota*peso_penalidade[2] +\
                  penalidade_CTI2_close_in*peso_penalidade[3] +\
                  penalidade_CTI2_80*peso_penalidade[4] +\
                  penalidade_CTI2_barra_remota*peso_penalidade[5]

#==============================================================================#


    return func_obj + penalidades

'''
#==============================================================================#
#==================================Otimizador==================================#
#==============================================================================#
'''

peso_penalidade = np.array([50, 50, 50, 50, 50, 50])
#abertura do arquivo de registro das execuções do script
now = datetime.now()
historico = open('historico_otimizacao_{}.txt'.format(now.strftime("%d%m%Y%H%M%S")), 'a')
historico.write("Inicio da execução em: {}\n\n".format(now.strftime("%H:%M:%S")))


for semente_otimizador in range(quantidade_de_sementes):
    #historico.write("\n#==========================Execução %d==========================#\n" %(semente_otimizador))
    np.random.seed(semente_otimizador)
    '''
    optimizer = ps.single.GlobalBestPSO(n_particles=numero_de_particulas,
                                        dimensions=quantidade_de_reles*3,
                                        options=options,
                                        bounds=bounds,
                                        bh_strategy='periodic',
                                        oh_strategy='exp_decay',
                                        velocity_clamp = (-0.7,0.7),
                                        vh_strategy = 'invert') #confirmar se não estou deixando ele pegar valores fora dos limites com essas estrategias
    '''
    options = {'c1': 2, 'c2': 2, 'w':0.4}
    optimizer = GlobalBestPSO(n_particles=numero_de_particulas, dimensions=quantidade_de_reles*3, options=options, bounds=bounds)

    cost, pos = optimizer.optimize(funcao_objetivo, iters=quantidade_de_iteracoes)


    print("\n\n")
    print("Custo Calculado = ",funcao_objetivo_verbose(pos))
    print("Custo do Otimizador = ",cost)

    #organização dos resultados
    contador = 0
    matriz_saida = np.zeros((quantidade_de_reles, 4))

    #separando passando os dados de um vetor [n] e tambem o tempo de operacao para o curto closein para uma matriz [n/2,3]
    for aux in range(quantidade_de_reles):
        matriz_saida[aux, 0] = pos[aux]  # primeira coluna apresenta os valores de TDS
        # segunda coluna apresenta os valores de Ipk
        matriz_saida[aux, 1] = pos[aux+quantidade_de_reles]
        # tempo de operacao dos RDS
        matriz_saida[aux, 2] = top(pos[aux], pos[aux+8], I_curto_close_in[aux], 1)
        # terceira coluna apresenta os valores de TRD
        matriz_saida[aux, 3] = pos[aux+2*quantidade_de_reles]
        contador = contador+1

    #===Apresentacao de Dados===#
    print("\n\n\n\n\n")
    #passando os dados da matriz para um dataframe
    dados_saida = {'Tempo[s]': matriz_saida[:, 2], 'TDS': matriz_saida[:, 0], 'Ipk[A]': matriz_saida[:, 1],
                   'TRD[s]': matriz_saida[:, 3]}

    #formatando o dataframe
    indice = []
    [indice.append(aux+1) for aux in range(quantidade_de_reles)]
    dataframe_saida = pd.DataFrame(data=dados_saida, index=indice)
    dataframe_saida['Tempo[s]'] = dataframe_saida['Tempo[s]'].map(
        '  {:3.5f}'.format)
    dataframe_saida['TDS'] = dataframe_saida['TDS'].map('  {:2.5f}'.format)
    dataframe_saida['Ipk[A]'] = dataframe_saida['Ipk[A]'].map('  {:4.2f}'.format)
    dataframe_saida['TRD[s]'] = dataframe_saida['TRD[s]'].map('  {:3.5f}'.format)
    print(tabulate(dataframe_saida, showindex=True, headers=dataframe_saida.columns))
    print("\n")

    #======PLOTAGEM DOS GRAFICOS======#

    #gerando espaço de correntes de 4000 ate 800 [A] para utilizar na plotagem
    correntes = np.linspace(4000, 800, num=43)
    tempos_atuacao = np.zeros((correntes.size,quantidade_de_reles))


    tempos_principal = np.zeros(correntes.size)
    tempos_backup = np.zeros(correntes.size)

#====conferindo coordenacao
    solucao_nao_satisfatoria = 0
    for i in range(Rele_principal.size):
      if (Rele_principal[i] != Rele_secundario[i]):
        RP = Rele_principal[i]
        RB = Rele_secundario[i]

        if (top(matriz_saida[RB-1,0], matriz_saida[RB-1,1],correntes[aux],1) < top(matriz_saida[RP-1,0], matriz_saida[RP-1,1],correntes[aux],1)):
          print("Relés %d e %d não estão coordenando!\n"%(RP, RB))
          solucao_nao_satisfatoria=1

    historico.write("{:.3f};".format(cost))
    for i in range(quantidade_de_reles*3):
        historico.write("{:.3f};".format(pos[i]))
    historico.write("\n")

    #com as correntes geradas, calculando o tempo de operação de cada rele
    for i in range(quantidade_de_reles):
      tempos_atuacao[:,i] = top(matriz_saida[i,0], matriz_saida[i,1],correntes,1)

    fig, axs = plt.subplots(10,1, figsize=[5,30])
    fig.suptitle("Coordenação Entre Principal e Backup")
    contador=0

    for i in range(Rele_principal.size):
      if (Rele_principal[i] != Rele_secundario[i]):
        axs[contador].loglog(correntes, tempos_atuacao[:,Rele_principal[i]-1], "k--",correntes, tempos_atuacao[:,Rele_secundario[i]-1], "k-")
        axs[contador].legend(['R{}'.format(Rele_principal[i]),'R{}'.format(Rele_secundario[i])], loc="upper right")
        contador+=1

    plt.show()
    #plotagem do custo histórico da otimizacao

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()


now = datetime.now()
historico.write("\nFim da execução em: {}".format(now.strftime("%H:%M:%S")))
historico.write("\n")
historico.close()
