#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define L 4  // número de linhas (deve ser divisível por P)
#define C 5  // número de colunas

// Matriz fixa de exemplo (4x5)
int matriz_fixa[L][C] = {
    {10, 20, 30, 40, 50},
    {5, 5, 5, 5, 5},
    {1, 3, 6, 10, 15},
    {100, 80, 60, 40, 20}
};

void print_matrix(const char* titulo, int* mat, int linhas, int colunas) {
    printf("\n%s\n", titulo);
    for (int i = 0; i < linhas; ++i) {
        for (int j = 0; j < colunas; ++j)
            printf("%4d", mat[i * colunas + j]);
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (L % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Erro: número de linhas deve ser divisível pelo número de processos.\n");
        MPI_Finalize();
        return 1;
    }

    int linhas_por_proc = L / size;
    int* local_linhas = (int*)malloc(linhas_por_proc * C * sizeof(int));
    int* local_grad = (int*)malloc(linhas_por_proc * (C - 1) * sizeof(int));
    int* gradiente_final = NULL;

    // Distribui a matriz
    MPI_Scatter(matriz_fixa, linhas_por_proc * C, MPI_INT,
                local_linhas, linhas_por_proc * C, MPI_INT,
                0, MPI_COMM_WORLD);

    // Calcula o gradiente local
    for (int i = 0; i < linhas_por_proc; ++i)
        for (int j = 0; j < C - 1; ++j)
            local_grad[i * (C - 1) + j] = local_linhas[i * C + j + 1] - local_linhas[i * C + j];

    // Processo 0 coleta os gradientes
    if (rank == 0)
        gradiente_final = (int*)malloc(L * (C - 1) * sizeof(int));

    MPI_Gather(local_grad, linhas_por_proc * (C - 1), MPI_INT,
               gradiente_final, linhas_por_proc * (C - 1), MPI_INT,
               0, MPI_COMM_WORLD);

    // Impressão dos resultados
    if (rank == 0) {
        printf("\nMatriz Original:\n");
        for (int i = 0; i < L; ++i) {
            for (int j = 0; j < C; ++j)
                printf("%4d", matriz_fixa[i][j]);
            printf("\n");
        }

        print_matrix("Gradiente Horizontal:", gradiente_final, L, C - 1);
        free(gradiente_final);
    }

    free(local_linhas);
    free(local_grad);
    MPI_Finalize();
    return 0;
}
