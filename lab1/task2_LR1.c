#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void run_task2(int n, int threads, const char* schedule_type, int chunk) {
    double* a = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));

    if (!a || !b) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
        b[i] = 0.0;
    }

    omp_set_num_threads(threads);
    double start_time = omp_get_wtime();

    if (strcmp(schedule_type, "static") == 0) {
        if (chunk > 0) {
#pragma omp parallel for schedule(static, chunk)
            for (int i = 1; i < n - 1; i++) {
                b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
            }
        }
        else {
#pragma omp parallel for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
            }
        }
    }
    else if (strcmp(schedule_type, "dynamic") == 0) {
        int ch = (chunk > 0) ? chunk : 1;
#pragma omp parallel for schedule(dynamic, ch)
        for (int i = 1; i < n - 1; i++) {
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
        }
    }
    else if (strcmp(schedule_type, "guided") == 0) {
        if (chunk > 0) {
#pragma omp parallel for schedule(guided, chunk)
            for (int i = 1; i < n - 1; i++) {
                b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
            }
        }
        else {
#pragma omp parallel for schedule(guided)
            for (int i = 1; i < n - 1; i++) {
                b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
            }
        }
    }
    else if (strcmp(schedule_type, "runtime") == 0) {
#pragma omp parallel for schedule(runtime)
        for (int i = 1; i < n - 1; i++) {
            b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0;
        }
    }
    else {
        printf("Unknown schedule type: %s\n", schedule_type);
        free(a);
        free(b);
        return;
    }

    double end_time = omp_get_wtime();

    printf("Schedule: %-8s (chunk: %-4d) | Threads: %d | N: %d | Time: %.6f sec | Check b[1]=%.2f, b[%d]=%.2f\n",
        schedule_type, chunk, threads, n, (end_time - start_time), b[1], n - 2, b[n - 2]);

    free(a);
    free(b);
}

int main(int argc, char* argv[]) {
    int n = 16000;
    int threads = 8;
    char schedule_type[32] = "all";
    int chunk = 0;

    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) strncpy(schedule_type, argv[3], sizeof(schedule_type) - 1);
    if (argc > 4) chunk = atoi(argv[4]);

    if (strcmp(schedule_type, "all") == 0) {
        printf("=== Testing OpenMP Work Schedules ===\n");
        run_task2(n, threads, "static", 0);
        run_task2(n, threads, "static", 100);
        run_task2(n, threads, "dynamic", 1);
        run_task2(n, threads, "dynamic", 100);
        run_task2(n, threads, "guided", 10);
        run_task2(n, threads, "runtime", 0);
    }
    else {
        run_task2(n, threads, schedule_type, chunk);
    }

    return 0;
}