#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Gera número aleatório usando uma semente local (Linear Congruential Generator, simples)
unsigned int lcg_rand(unsigned int *seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return *seed;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Uso: %s <tamanho do vetor> <número de threads>\n", argv[0]);
        return 1;
    }

    int vectorQuantity = atoi(argv[1]);
    int threadNumber = atoi(argv[2]);

    int *randomNumbers = (int *)malloc(vectorQuantity * sizeof(int));

    #pragma omp parallel if(threadNumber > 0) num_threads(threadNumber) shared(randomNumbers, vectorQuantity)
    {
        int tid = omp_get_thread_num();
        // Semente única baseada no tempo + id da thread
        unsigned int seed = (unsigned int)time(NULL) ^ (tid * 1234567);

        #pragma omp for
        for (int i = 0; i < vectorQuantity; i++) {
            int number = lcg_rand(&seed) % 100;
            printf("Thread ID: %d | Numero: %d\n", tid, number);
            randomNumbers[i] = number;
        }
    }

    printf("Random Numbers: ");
    for (int i = 0; i < vectorQuantity; i++) {
        printf("%d ", randomNumbers[i]);
    }
    printf("\n");

    free(randomNumbers);
    return 0;
}
