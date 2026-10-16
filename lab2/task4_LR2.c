#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* Dynamic memory allocation for an N x N matrix */
double** allocate_matrix(int n) {
    double** mat = (double**)malloc(n * sizeof(double*));
    if (!mat) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        mat[i] = (double*)malloc(n * sizeof(double));
        if (!mat[i]) {
            fprintf(stderr, "Error: Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }
    }
    return mat;
}

/* Free allocated matrix memory */
void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

/* Fill matrix with random values in user-defined range [min_val, max_val] */
void fill_matrix_random(double** mat, int n, double min_val, double max_val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double scale = (double)rand() / (double)RAND_MAX;
            mat[i][j] = min_val + scale * (max_val - min_val);
        }
    }
}

/* Fill matrix manually from standard input */
void fill_matrix_manual(double** mat, int n, const char* name) {
    printf("Enter elements for matrix %s (%d x %d):\n", name, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%s[%d][%d] = ", name, i, j);
            if (scanf_s("%lf", &mat[i][j]) != 1) {
                fprintf(stderr, "Error: Invalid numeric input.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
}

/* Print matrix to console (used for small matrices) */
void print_matrix(double** mat, int n, const char* name) {
    printf("\nMatrix %s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f ", mat[i][j]);
        }
        printf("\n");
    }
}

/* Sequential matrix multiplication */
void multiply_sequential(double** a, double** b, double** c, int n) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

/* Parallel matrix multiplication using OpenMP */
void multiply_parallel(double** a, double** b, double** c, int n) {
    int i, j, k;
    double sum;

#pragma omp parallel for shared(a, b, c, n) private(i, j, k, sum)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

/* Verify that sequential and parallel results match */
int verify_results(double** c_seq, double** c_par, int n) {
    const double eps = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(c_seq[i][j] - c_par[i][j]) > eps) {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int n = 0;
    int num_threads = 0;
    int mode = 1;
    double min_val = 0.0, max_val = 10.0;

    srand((unsigned int)time(NULL));

    /* Read N and thread count from CLI args if provided, otherwise prompt user */
    if (argc >= 3) {
        n = atoi(argv[1]);
        num_threads = atoi(argv[2]);
    }
    else {
        printf("Enter matrix size N: ");
        if (scanf_s("%d", &n) != 1) {
            fprintf(stderr, "Error: Invalid input for N.\n");
            return EXIT_FAILURE;
        }
        printf("Enter number of threads: ");
        if (scanf_s("%d", &num_threads) != 1) {
            fprintf(stderr, "Error: Invalid input for number of threads.\n");
            return EXIT_FAILURE;
        }
    }

    if (n <= 0 || num_threads <= 0) {
        fprintf(stderr, "Error: Matrix size and number of threads must be positive integers.\n");
        return EXIT_FAILURE;
    }

    double** a = allocate_matrix(n);
    double** b = allocate_matrix(n);
    double** c_seq = allocate_matrix(n);
    double** c_par = allocate_matrix(n);

    /* If min and max values are passed as 3rd and 4th CLI args, use them directly */
    if (argc >= 5) {
        min_val = atof(argv[3]);
        max_val = atof(argv[4]);
        fill_matrix_random(a, n, min_val, max_val);
        fill_matrix_random(b, n, min_val, max_val);
    }
    else {
        printf("Select input mode (1 - Custom range generation, 2 - Manual element input): ");
        if (scanf_s("%d", &mode) != 1 || (mode != 1 && mode != 2)) {
            fprintf(stderr, "Error: Invalid mode selected.\n");
            return EXIT_FAILURE;
        }

        if (mode == 1) {
            printf("Enter min and max values for matrix elements (e.g., 1 100): ");
            if (scanf_s("%lf %lf", &min_val, &max_val) != 2 || min_val > max_val) {
                fprintf(stderr, "Error: Invalid range.\n");
                return EXIT_FAILURE;
            }
            fill_matrix_random(a, n, min_val, max_val);
            fill_matrix_random(b, n, min_val, max_val);
        }
        else {
            fill_matrix_manual(a, n, "A");
            fill_matrix_manual(b, n, "B");
        }
    }

    omp_set_num_threads(num_threads);

    printf("\n==================================================\n");
    printf("Matrix Multiplication Benchmark (Task 4 / Lab 2)\n");
    printf("Matrix size          : %d x %d\n", n, n);
    printf("Available processors : %d\n", omp_get_num_procs());
    printf("Requested threads    : %d\n", num_threads);
    if (mode == 1) {
        printf("Value range          : [%.2f, %.2f]\n", min_val, max_val);
    }
    else {
        printf("Input mode           : Manual\n");
    }
    printf("==================================================\n");

    /* 1. Sequential execution */
    double start_seq = omp_get_wtime();
    multiply_sequential(a, b, c_seq, n);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    printf("Sequential execution time        : %.6f sec\n", time_seq);

    /* 2. Parallel execution */
    double start_par = omp_get_wtime();
    multiply_parallel(a, b, c_par, n);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    printf("Parallel execution time (%d thr.) : %.6f sec\n", num_threads, time_par);

    /* 3. Speedup and verification */
    double speedup = time_seq / time_par;
    int is_correct = verify_results(c_seq, c_par, n);

    printf("--------------------------------------------------\n");
    printf("Speedup                          : %.2fx\n", speedup);
    printf("Verification                     : %s\n", is_correct ? "PASSED (OK)" : "FAILED");
    printf("==================================================\n");

    /* Print matrices if manual input was used and N is small */
    if (mode == 2 && n <= 10) {
        print_matrix(a, n, "A");
        print_matrix(b, n, "B");
        print_matrix(c_par, n, "C (Result)");
    }

    free_matrix(a, n);
    free_matrix(b, n);
    free_matrix(c_seq, n);
    free_matrix(c_par, n);

    return EXIT_SUCCESS;
}