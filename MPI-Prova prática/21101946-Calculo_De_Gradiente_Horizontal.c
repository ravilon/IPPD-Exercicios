#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*
Nome: Rávilon Aguiar Dos Santos
Matrícula: 21101946
*/

#define L 4  
#define C 5  

// Matriz fixa para o exemplo
int matriz[L][C] = {
    {10, 20, 30, 40, 50},
    {5, 5, 5, 5, 5},
    {1, 3, 6, 10, 15},
    {100, 80, 60, 40, 20}
};

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (L % size != 0) {
        if (rank == 0) printf("O número de linhas não é divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    int linhas_por_processo = L / size;

    // Aloca espaço para as linhas locais
    int linhas_locais[linhas_por_processo][C];
    int gradiente_local[linhas_por_processo][C - 1];

    // Distribui as linhas da matriz original para todos os processos
    MPI_Scatter(matriz, linhas_por_processo * C, MPI_INT,
                linhas_locais, linhas_por_processo * C, MPI_INT,
                0, MPI_COMM_WORLD);

    // Cada processo calcula o gradiente horizontal local
    for (int i = 0; i < linhas_por_processo; i++) {
        for (int j = 0; j < C - 1; j++) {
            gradiente_local[i][j] = linhas_locais[i][j + 1] - linhas_locais[i][j];
        }
    }

    // Processo 0 reúne os gradientes locais
    int gradiente_final[L][C - 1];
    MPI_Gather(gradiente_local, linhas_por_processo * (C - 1), MPI_INT,
               gradiente_final, linhas_por_processo * (C - 1), MPI_INT,
               0, MPI_COMM_WORLD);

    // Processo 0 imprime a matriz original e a matriz gradiente
    if (rank == 0) {
        printf("\nMatriz Original:\n");
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < C; j++)
                printf("%4d", matriz[i][j]);
            printf("\n");
        }

        printf("\nGradiente Horizontal:\n");
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < C - 1; j++)
                printf("%4d", gradiente_final[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}