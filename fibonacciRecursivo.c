#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
// Compile with: gcc -fopenmp fibonacciRecursivo.c -o fibonacciRecursivo
int fib(int n) {
    int r1, r2;
    if (n<2) return n;
#pragma omp task shared(r1) firstprivate(n)
    r1=fib(n-1);
#pragma omp task shared(r2) firstprivate(n)
    r2=fib(n-2);
#pragma omp taskwait
    return r1+r2;
}

int main( int argc, char **argv ) {
    int r, n = atoi(argv[1]);
    double start, end;
    
    start = omp_get_wtime();
    #pragma omp parallel
    
    {
        #pragma omp single

        printf("thread: %d",omp_get_thread_num());
        r = fib(n);
    }
    
    end = omp_get_wtime();
    
    printf("- %d -\n",r);
    printf("Tempo: %4.2f\n",end-start);
    return 0;
}

