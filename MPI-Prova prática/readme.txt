Atividade Prática — MPI: Cálculo do Gradiente Horizontal de uma Matriz
Objetivo
Implementar um programa MPI que distribua linhas de uma matriz entre os processos e calcule o gradiente horizontal de cada linha. A atividade exercita o uso de comunicação coletiva e o processamento paralelo de dados estruturados em grade.

Introdução ao Problema
Em várias aplicações científicas e computacionais, é comum trabalhar com matrizes que representam grandezas físicas ou sinais digitais. Nessas situações, é útil estimar como os valores variam no espaço, o que pode ser feito por meio do gradiente.

Neste exercício, vamos trabalhar com o gradiente horizontal discreto, que mede a variação entre valores consecutivos de uma mesma linha da matriz. Esse tipo de operação é bastante comum em:

Processamento de imagens, para detecção de bordas (como no operador de Sobel);

Simulações físicas, para estimar variações de temperatura, pressão etc.;

Análise de dados geográficos, como elevações em mapas digitais.

Definição Matemática do Gradiente
Seja uma matriz M de dimensão L x C. Para cada linha i da matriz, definimos o gradiente horizontal discreto como:

G parêntese recto esquerdo i vírgula j parêntese recto direito igual a M parêntese recto esquerdo i vírgula j mais 1 parêntese recto direito menos M parêntese recto esquerdo i vírgula j parêntese recto direito espaço em branco
para 
0
≤
j
<
C
−
1
G[i, j] = M[i, j+1] - M[i, j] \quad \text{para } 0 \leq j < C - 1
G[i,j]=M[i,j+1]−M[i,j]para 0≤j<C−1

Ou seja, o elemento $G[i, j]$ mede quanto o valor da célula muda da posição $j$ para a posição $j+1$.

Como consequência, a matriz gradiente G terá dimensão L x (C - 1).

Especificação da Atividade
Você deve implementar um programa MPI que:

No processo 0:

Cria uma matriz M[L][C] de números inteiros ou reais (por exemplo, valores entre 0 e 100).

A matriz pode ser gerada com valores fixos ou aleatórios.

Distribui as linhas da matriz igualmente entre os processos.

Cada processo calcula o gradiente horizontal local das suas linhas:

Para cada linha local, compute M[i][j+1] - M[i][j] para j = 0 até C - 2.

Os resultados (gradientes locais) são reunidos no processo 0.

O processo 0 imprime (ou salva) a matriz gradiente G.

Requisitos e Restrições
O número de linhas L deve ser divisível pelo número de processos P (isto é, L mod P = 0).

Cada processo manipula um bloco contínuo de L/P linhas.

Utilize tipos de dados simples (inteiros ou ponto flutuante).

O cálculo do gradiente é somente horizontal; o vertical não será considerado.

Exemplo do cálculo do Gradiente da Linha
Em Python. Veja a imagem original e o significado do cálculo do gradiente.

import math

# Parâmetros
TAMANHO = 15  # matriz TAMANHO x TAMANHO
RAIO = TAMANHO // 3

# Cria matriz com círculo centrado
def cria_matriz_com_circulo(tamanho, raio):
    centro = tamanho // 2
    matriz = []
    for i in range(tamanho):
        linha = []
        for j in range(tamanho):
            dist = math.sqrt((i - centro) ** 2 + (j - centro) ** 2)
            if dist <= raio:
                linha.append(100)
            else:
                linha.append(0)
        matriz.append(linha)
    return matriz

# Gradiente horizontal: G[i][j] = M[i][j+1] - M[i][j]
def gradiente_horizontal(matriz):
    linhas = len(matriz)
    colunas = len(matriz[0])
    grad = []
    for i in range(linhas):
        linha = []
        for j in range(colunas - 1):
            linha.append(matriz[i][j+1] - matriz[i][j])
        grad.append(linha)
    return grad

# Impressão numérica formatada
def imprime_matriz(matriz, titulo):
    print(f"\n{titulo}")
    for linha in matriz:
        print(" ".join(f"{v:4d}" for v in linha))

# Execução principal
matriz = cria_matriz_com_circulo(TAMANHO, RAIO)
grad = gradiente_horizontal(matriz)

imprime_matriz(matriz, "Matriz binária com círculo:")
imprime_matriz(grad, "Gradiente horizontal:")

