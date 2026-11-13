#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef struct {
    double x;
    double y;
    double z;
} Point3D;

// Генерация файла с координатами точек, чтобы не создавать вручную
void generate_points_file(const char* filename, int n) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to create points file");
        exit(1);
    }

    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++) {
        // Детерминированные координаты, чтобы легко проверить среднее аналитически
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

// 0. Последовательная версия 
void calc_center_serial(const Point3D* pts, int n, Point3D* center, double* elapsed) {
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    double t_start = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        sum_x += pts[i].x;
        sum_y += pts[i].y;
        sum_z += pts[i].z;
    }

    center->x = sum_x / n;
    center->y = sum_y / n;
    center->z = sum_z / n;

    *elapsed = omp_get_wtime() - t_start;
}

// 1. Версия с распараллеливанием цикла 
void calc_center_parallel_loop(const Point3D* pts, int n, int threads, Point3D* center, double* elapsed) {
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int i;

    omp_set_num_threads(threads);
    double t_start = omp_get_wtime();

#pragma omp parallel for private(i) shared(pts, n) reduction(+:sum_x, sum_y, sum_z)
    for (i = 0; i < n; i++) {
        sum_x += pts[i].x;
        sum_y += pts[i].y;
        sum_z += pts[i].z;
    }

    center->x = sum_x / n;
    center->y = sum_y / n;
    center->z = sum_z / n;

    *elapsed = omp_get_wtime() - t_start;
}

// 2. Версия с функциональной декомпозицией 
void calc_center_functional(const Point3D* pts, int n, int threads, Point3D* center, double* elapsed) {
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    omp_set_num_threads(threads);
    double t_start = omp_get_wtime();

#pragma omp parallel sections shared(pts, n, sum_x, sum_y, sum_z)
    {
#pragma omp section
        {
            double local_x = 0.0;
            for (int i = 0; i < n; i++) {
                local_x += pts[i].x;
            }
            sum_x = local_x;
        }

#pragma omp section
        {
            double local_y = 0.0;
            for (int i = 0; i < n; i++) {
                local_y += pts[i].y;
            }
            sum_y = local_y;
        }

#pragma omp section
        {
            double local_z = 0.0;
            for (int i = 0; i < n; i++) {
                local_z += pts[i].z;
            }
            sum_z = local_z;
        }
    }

    center->x = sum_x / n;
    center->y = sum_y / n;
    center->z = sum_z / n;

    *elapsed = omp_get_wtime() - t_start;
}

int main(int argc, char* argv[]) {
    int threads = 4;
    int n_points = 1000000;
    const char* filename = "points.txt";

    // Аргументы командной строки: task5_LR3.exe [threads] [N] [filename]
    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) n_points = atoi(argv[2]);
    if (argc > 3) filename = argv[3];

    if (threads <= 0) threads = 4;
    if (n_points <= 0) n_points = 1000000;

    printf("=== Lab 3 (Task 5): 3D Geometric Center ===\n");
    printf("Requested threads: %d | Target points N: %d | File: %s\n", threads, n_points, filename);

    // Проверяем файл; если его нет или размер не совпадает с переданным N — пересоздаём
    int file_n = 0;
    Point3D* pts = read_points_file(filename, &file_n);
    if (!pts || file_n != n_points) {
        if (pts) free(pts);
        printf("Generating '%s' with %d points...\n", filename, n_points);
        generate_points_file(filename, n_points);
        pts = read_points_file(filename, &file_n);
    }

    printf("Loaded %d points from '%s'.\n\n", file_n, filename);

    Point3D c_serial, c_loop, c_func;
    double t_serial = 0.0, t_loop = 0.0, t_func = 0.0;

    calc_center_serial(pts, file_n, &c_serial, &t_serial);
    calc_center_parallel_loop(pts, file_n, threads, &c_loop, &t_loop);
    calc_center_functional(pts, file_n, threads, &c_func, &t_func);

    printf("%-28s | Center: (%.2f, %.2f, %.2f) | Time: %.6f s | Speedup: %.2fx\n",
        "0. Serial (1 thread)", c_serial.x, c_serial.y, c_serial.z, t_serial, 1.0);

    printf("%-28s | Center: (%.2f, %.2f, %.2f) | Time: %.6f s | Speedup: %.2fx\n",
        "1. Data Parallel (for)", c_loop.x, c_loop.y, c_loop.z, t_loop,
        (t_loop > 0) ? t_serial / t_loop : 0.0);

    printf("%-28s | Center: (%.2f, %.2f, %.2f) | Time: %.6f s | Speedup: %.2fx\n",
        "2. Functional (sections)", c_func.x, c_func.y, c_func.z, t_func,
        (t_func > 0) ? t_serial / t_func : 0.0);

    free(pts);
    return 0;
}

