#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    double x;
    double y;
    double z;
} Point3D;

// Генерация тестового файла с координатами точек
void generate_points_file(const char* filename, int n) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to create points file");
        exit(1);
    }

    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++) {
        // x_i = i + 1, y_i = 2*(i + 1), z_i = 3*(i + 1)
        // Среднее по x, y, z будет равно: ( (N+1)/2 + 2*(N+1)/2 + 3*(N+1)/2 ) / 3 = N + 1
        double x = (double)(i + 1) * 1.0;
        double y = (double)(i + 1) * 2.0;
        double z = (double)(i + 1) * 3.0;
        fprintf(f, "%.4f %.4f %.4f\n", x, y, z);
    }
    fclose(f);
}

// Чтение массива 3D-точек из файла
Point3D* read_points_file(const char* filename, int* out_n) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        return NULL;
    }

    int n = 0;
    if (fscanf(f, "%d", &n) != 1 || n <= 0) {
        fclose(f);
        return NULL;
    }

    Point3D* pts = (Point3D*)malloc((size_t)n * sizeof(Point3D));
    if (!pts) {
        fclose(f);
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%lf %lf %lf", &pts[i].x, &pts[i].y, &pts[i].z) != 3) {
            fprintf(stderr, "File format error at line %d\n", i + 2);
            free(pts);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
    *out_n = n;
    return pts;
}

// 0. Последовательная версия (эталон для сверки точности и времени)
double calc_serial(const Point3D* pts, int n, double* elapsed) {
    double global_sum = 0.0;
    double t_start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        global_sum += pts[i].x + pts[i].y + pts[i].z;
    }

    double result = global_sum / (3.0 * (double)n);
    *elapsed = omp_get_wtime() - t_start;
    return result;
}

// 1. Функциональная декомпозиция + глобальная сумма + критические секции
double calc_functional_critical(const Point3D* pts, int n, int threads, double* elapsed) {
    double global_sum = 0.0; // Глобальная (разделяемая) сумма

    omp_set_num_threads(threads);
    double t_start = omp_get_wtime();

#pragma omp parallel sections shared(pts, n, global_sum)
    {
        // Секция 1: подсчет суммы по координате X
#pragma omp section
        {
            double local_sum_x = 0.0;
            for (int i = 0; i < n; i++) {
                local_sum_x += pts[i].x;
            }
#pragma omp critical
            {
                global_sum += local_sum_x;
            }
        }

        // Секция 2: подсчет суммы по координате Y
#pragma omp section
        {
            double local_sum_y = 0.0;
            for (int i = 0; i < n; i++) {
                local_sum_y += pts[i].y;
            }
#pragma omp critical
            {
                global_sum += local_sum_y;
            }
        }

        // Секция 3: подсчет суммы по координате Z
#pragma omp section
        {
            double local_sum_z = 0.0;
            for (int i = 0; i < n; i++) {
                local_sum_z += pts[i].z;
            }
#pragma omp critical
            {
                global_sum += local_sum_z;
            }
        }
    }

    double result = global_sum / (3.0 * (double)n);
    *elapsed = omp_get_wtime() - t_start;
    return result;
}

int main(int argc, char* argv[]) {
    int threads = 4;
    int n_points = 5000000;
    const char* filename = "points.txt";

    // Аргументы командной строки: task6_LR4.exe [threads] [N] [filename]
    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) n_points = atoi(argv[2]);
    if (argc > 3) filename = argv[3];

    if (threads <= 0) threads = 4;
    if (n_points <= 0) n_points = 5000000;

    printf("=== Lab 4 (Task 6): Functional Decomposition & Critical Sections ===\n");
    printf("Requested threads: %d | Target points N: %d | File: %s\n", threads, n_points, filename);

    int file_n = 0;
    Point3D* pts = read_points_file(filename, &file_n);
    if (!pts || file_n != n_points) {
        if (pts) free(pts);
        printf("Generating '%s' with %d points...\n", filename, n_points);
        generate_points_file(filename, n_points);
        pts = read_points_file(filename, &file_n);
    }

    printf("Loaded %d points from '%s'.\n\n", file_n, filename);

    double t_serial = 0.0, t_func = 0.0;
    double res_serial = calc_serial(pts, file_n, &t_serial);
    double res_func = calc_functional_critical(pts, file_n, threads, &t_func);

    printf("%-32s | Result: %.4f | Time: %.6f s | Speedup: %.2fx\n",
        "0. Serial (1 thread)", res_serial, t_serial, 1.0);

    printf("%-32s | Result: %.4f | Time: %.6f s | Speedup: %.2fx\n",
        "1. Functional + Critical", res_func, t_func,
        (t_func > 0.0) ? t_serial / t_func : 0.0);

    free(pts);
    return 0;
}