#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <windows.h>

// Способ 1: Spin-wait с задержкой и flush
void method1_spin_wait(int threads) {
    printf("\n--- Method 1: Spin-wait with #pragma omp flush ---\n");
    volatile int current = threads - 1;

#pragma omp parallel shared(current)
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        while (1) {
#pragma omp flush(current)
            if (current == tid) {
#pragma omp critical
                {
                    printf("Hello World from thread %d of %d\n", tid, total);
                    current--;
#pragma omp flush(current)
                }
                break;
            }
            Sleep(1);
        }
    }
}

// Способ 2: Итеративный барьерный цикл (#pragma omp barrier)
void method2_barrier_loop(int threads) {
    printf("\n--- Method 2: Iterative barrier (#pragma omp barrier) ---\n");

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();

        for (int step = total - 1; step >= 0; step--) {
            if (tid == step) {
                printf("Hello World from thread %d of %d\n", tid, total);
            }
#pragma omp barrier
        }
    }
}

// Способ 3: Директива упорядочивания (#pragma omp ordered)
void method3_ordered(int threads) {
    printf("\n--- Method 3: Ordered directive (#pragma omp ordered) ---\n");

#pragma omp parallel
    {
#pragma omp for ordered schedule(static, 1)
        for (int i = 0; i < threads; i++) {
#pragma omp ordered
            {
                int target_tid = threads - 1 - i;
                printf("Hello World from thread %d (target ID: %d of %d)\n",
                    omp_get_thread_num(), target_tid, omp_get_num_threads());
            }
        }
    }
}

// Способ 4: Корректное использование замков (omp_lock_t) с передачей готовности
void method4_locks(int threads) {
    printf("\n--- Method 4: Lock-protected turn indicator (omp_lock_t API) ---\n");

    omp_lock_t lock;
    omp_init_lock(&lock);
    int current_turn = threads - 1;

#pragma omp parallel shared(lock, current_turn)
    {
        int tid = omp_get_thread_num();
        int total = omp_get_num_threads();
        int done = 0;

        while (!done) {
            omp_set_lock(&lock);
            if (current_turn == tid) {
                printf("Hello World from thread %d of %d\n", tid, total);
                current_turn--;
                done = 1;
            }
            omp_unset_lock(&lock);

            if (!done) {
                Sleep(1);
            }
        }
    }

    omp_destroy_lock(&lock);
}

// Способ 5: Параллельная запись в общий массив и обратный вывод
void method5_buffer(int threads) {
    printf("\n--- Method 5: Buffer collection & reverse print ---\n");

    int* buffer = (int*)malloc(threads * sizeof(int));
    if (!buffer) return;

#pragma omp parallel shared(buffer)
    {
        int tid = omp_get_thread_num();
        buffer[tid] = tid;
    }

    for (int i = threads - 1; i >= 0; i--) {
        printf("Hello World from thread %d of %d (from buffer)\n", buffer[i], threads);
    }

    free(buffer);
}

int main(int argc, char* argv[]) {
    int threads = 8;
    int method = 0;

    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) method = atoi(argv[2]);

    omp_set_num_threads(threads);

    printf("=== Testing 5 Methods to Print Thread IDs in Reverse Order ===\n");

    switch (method) {
    case 1: method1_spin_wait(threads); break;
    case 2: method2_barrier_loop(threads); break;
    case 3: method3_ordered(threads); break;
    case 4: method4_locks(threads); break;
    case 5: method5_buffer(threads); break;
    default:
        method1_spin_wait(threads);
        method2_barrier_loop(threads);
        method3_ordered(threads);
        method4_locks(threads);
        method5_buffer(threads);
        break;
    }

    return 0;
}