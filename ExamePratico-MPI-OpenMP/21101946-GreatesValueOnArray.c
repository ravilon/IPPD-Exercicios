#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Este programa encontra o maior valor em um vetor de 100 000
 * inteiros sem sinal utilizando MPI e OpenMP juntos. A ideia é
 * modularizar o código em algumas funções simples para facilitar a
 * leitura. A implementação não é profissional, apenas mostra uma
 * estrutura que um estudante poderia usar.
 */

 //mpicc -O2 -fopenmp "21101946-GreatesValueOnArray.c" -o 21101946-GreatesValueOnArray 
 //mpirun -np 4 ./21101946-GreatesValueOnArray

// Cria e preenche o vetor apenas no processo 0
static unsigned int *criar_e_preencher_vetor(int vectorsize, int rank) {
    unsigned int *vector = NULL;
    if (rank == 0) {
        vector = (unsigned int *)malloc(vectorsize * sizeof(unsigned int));
        if (vector == NULL) {
            return NULL;
        }
        srand((unsigned int)time(NULL));
        for (int i = 0; i < vectorsize; i++) {
            vector[i] = (unsigned int)rand();
        }
    }
    return vector;
}

// quantos elementos cada processo irá receber
static void calcular_envios(int total, int size, int *sendcounts, int *displs) {
    int base = total / size;
    int resto = total % size;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base;
        if (i < resto) {
            sendcounts[i]++;
        }
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }
    }
}

static unsigned int max_local(unsigned int *vector, int n) {
    unsigned int max_value = 0;
    
    #pragma omp parallel for reduction(max:max_value)
    for (int i = 0; i < n; i++) {
        if (vector[i] > max_value) {
            max_value = vector[i];
        }
    }
    return max_value;
}

static unsigned int reduzir_max(unsigned int local_max, int rank) {
    unsigned int global_max = 0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_UNSIGNED, MPI_MAX, 0,
               MPI_COMM_WORLD);
    return global_max;
}

int main(int argc, char *argv[]) {
    const int vectorsize = 100000; 
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Vetores de contagem e deslocamento para MPI_Scatterv */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    if (!sendcounts || !displs) {
        fprintf(stderr, "Erro de alocação\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Calcula quantos elementos cada processo irá receber */
    calcular_envios(vectorsize, size, sendcounts, displs);

    /* Aloca vetor local */
    int local_n = sendcounts[rank];
    unsigned int *local_vec = (unsigned int *)malloc(local_n * sizeof(unsigned int));
    if (!local_vec) {
        fprintf(stderr, "Falha ao alocar vetor local no processo %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Cria e preenche o vetor apenas no processo 0 */
    unsigned int *vetor_global = criar_e_preencher_vetor(vectorsize, rank);
    if (rank == 0 && vetor_global == NULL) {
        fprintf(stderr, "Falha ao alocar o vetor principal\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Distribui partes do vetor para todos os processos */
    MPI_Scatterv(vetor_global, sendcounts, displs, MPI_UNSIGNED,
                 local_vec, local_n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    /* Calcula o maior valor do pedaço local */
    unsigned int local_m = max_local(local_vec, local_n);

    /* Reduz para obter o maior valor global no processo 0 */
    unsigned int global_m = reduzir_max(local_m, rank);

    if (rank == 0) {
        printf("Maior valor encontrado: %u\n", global_m);
    }

    /* Libera memórias */
    free(local_vec);
    free(sendcounts);
    free(displs);
    if (vetor_global) {
        free(vetor_global);
    }

    MPI_Finalize();
    return 0;
}