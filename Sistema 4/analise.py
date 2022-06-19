import numpy as np
import statistics
#leitura
arquivo = open("historico_otimizacao.csv")
resultados = np.loadtxt(arquivo, delimiter=";")
arquivo.close()

outputfile = open('resultado_estat.txt', 'a')

quantidade_de_sementes = len(resultados[:,1])
print("Quantidade de sementes executadas: \n", quantidade_de_sementes)
outputfile.write("Quantidade de sementes executadas: {}\n".format(quantidade_de_sementes))
minimo = np.min(resultados[:,0])
print("Menor somatório obtido: ",minimo)
outputfile.write("Menor somatório obtido: {}\n".format(minimo))
posicao_minimo = [i for i, x in enumerate(resultados[:,0]) if x == minimo] #esse vetor guarda todas as vezes que o somatorio minimo aparece na lista de resultados

print("Solução do menor somatório",posicao_minimo[0])
print(resultados[posicao_minimo[0],:])
print("Porcentagem de vezes que o menor somatório aparece: %d%%"%(100*len(posicao_minimo)/quantidade_de_sementes))
outputfile.write("Porcentagem de vezes que o menor somatório aparece: {}%\n".format(100*len(posicao_minimo)/quantidade_de_sementes))


medias = np.zeros(len(resultados[1,:]))
desvios = np.zeros(len(resultados[1,:]))

for i in range(len(resultados[1,:])): #porque sao 4 linhas
    medias[i] = statistics.mean(resultados[:,i])
    desvios[i] = statistics.pstdev(resultados[:,i])

print("Média dos resultados")
outputfile.write("Variável & Média & Desvio Padrão \\\\\n ")
for i in range(len(resultados[1,:])):
    outputfile.write("{:d} & {:.3f} & {:.3f}\\\\\n ".format(i, medias[i], desvios[i]))

print(medias)
print("\nDesvios padrão dos resultados\n")

outputfile.write("\nDesvios padrão dos resultados\n")
print(desvios)

outputfile.close()
