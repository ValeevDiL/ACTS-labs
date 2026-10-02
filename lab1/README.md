# Лабораторная работа №1 (OpenMP)

Дисциплина: Аппаратные средства телекоммуникационных систем
Группа: 4404

## Описание файлов
- task1_LR1.c — вывод Hello World с ID потока на 8 потоках.
- task2_LR1.c — сглаживание массива (16 000 элементов) с типами расписаний static, dynamic, guided, runtime.
- task3_LR1.c — 5 способов вывода номеров потоков в обратном порядке.

## Сборка программы (GCC / MinGW)
```bash
gcc -fopenmp task1_LR1.c -o task1_LR1
gcc -fopenmp task2_LR1.c -o task2_LR1
gcc -fopenmp task3_LR1.c -o task3_LR1

Команды для запуска:

./task1_LR1 8
./task2_LR1 8 16000 all
./task3_LR1 8 0