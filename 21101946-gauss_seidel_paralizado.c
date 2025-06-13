#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 100
#define MAX_IT 1000
#define TOL 1e-6

/*
Matrícula: 21101946
Aluno: Rávilon Aguiar Dos Santos

Após a execução de ambas versões observei que a paralelizada demorava em torno de 
~0.012s com 1 thread,
~0.135s com 4 threads, 
~0.252s com 8 threads  
enquanto a versão sequencial demorou 
~0.001s

Nesse caso para um número pequeno de pontos a versão paralelizada
não se mostrou vantajosa, mas para um número maior de pontos a versão paralelizada se torna mais vantajosa.

Observação: Notei também uma ligeira diferença nos valores finais obtidos entre as versões. 
Isso ocorre porque a versão paralela não utiliza imediatamente os valores recém-atualizados, 
diferete da sequencial, resultando em pequenas variações devido à ordem e dependência das atualizações.
*/


/*
     * Paralelização:
     * - As atualizações dos pontos "vermelhos" (índices pares) e "pretos" (índices ímpares) do vetor u
     *   são feitas em paralelo com diretivas `#pragma omp parallel for`.
     * - O cálculo da diferença máxima (max_diff) também é paralelizado usando a cláusula `reduction(max:max_diff)`.
     * - Para evitar condições de corrida, os novos valores são primeiro armazenados em vetores temporários (temp_red e temp_black),
     *   que depois são copiados para o vetor u em uma segunda passagem paralela.
*/
void gauss_seidel_rb(double* u, int n, int max_it, double tol) {
     
    for (int it = 0; it < max_it; it++) {
        double max_diff = 0.0;

        // Red update: even indices
        double* temp_red = (double*)malloc(n * sizeof(double));
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            temp_red[i] = 0.5 * (u[i - 1] + u[i + 1]);
        }
        #pragma omp parallel for reduction(max:max_diff)
        for (int i = 1; i < n - 1; i += 2) {
            double old = u[i];
            u[i] = temp_red[i];
            double diff = fabs(u[i] - old);
            if (diff > max_diff)
                max_diff = diff;
        }
        free(temp_red);

        // Black update: odd indices
        double* temp_black = (double*)malloc(n * sizeof(double));
        #pragma omp parallel for
        for (int i = 2; i < n - 1; i += 2) {
            temp_black[i] = 0.5 * (u[i - 1] + u[i + 1]);
        }
        #pragma omp parallel for reduction(max:max_diff)
        for (int i = 2; i < n - 1; i += 2) {
            double old = u[i];
            u[i] = temp_black[i];
            double diff = fabs(u[i] - old);
            if (diff > max_diff)
                max_diff = diff;
        }
        free(temp_black);

        if (max_diff < tol) {
            printf("Convergência alcançada na iteração %d\n", it);
            break;
        }
    }
}

int main() {
    int n_threads = 4;
    omp_set_num_threads(n_threads);

    double* u = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        u[i] = 0.0;
    }
    u[0] = 100.0;
    u[N - 1] = 50.0;

    double start = omp_get_wtime();
    gauss_seidel_rb(u, N, MAX_IT, TOL);
    double end = omp_get_wtime();

    printf("Tempo de execucao: %.6f segundos\n", end - start);

    for (int i = 0; i < N; i++) {
        printf("u[%d] = %.4f\n", i, u[i]);
    }

    free(u);
    return 0;
}

