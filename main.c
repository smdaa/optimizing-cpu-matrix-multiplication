#include <assert.h>
#include <cblas.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*gemm_function)(int, int, int, double *, double *, double *);

int read_matrix(const char *filename, double **matrix, int *rows, int *cols) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    return -1;
  }

  if (fscanf(file, "%d %d", rows, cols) != 2) {
    fclose(file);
    return -1;
  }

  *matrix = (double *)malloc((*rows) * (*cols) * sizeof(double));
  if (*matrix == NULL) {
    fclose(file);
    return -1;
  }

  for (int i = 0; i < *rows * *cols; i++) {
    if (fscanf(file, "%lf", &(*matrix)[i]) != 1) {
      fclose(file);
      free(*matrix);
      return -1;
    }
  }

  fclose(file);
  return 0;
}

int matrices_are_equal(int rows, int cols, double *matrix1, double *matrix2,
                       double tolerance) {
  for (int i = 0; i < rows * cols; i++) {
    if (fabs(matrix1[i] - matrix2[i]) > tolerance) {
      return 0;
    }
  }
  return 1;
}

void benchmark_gemm(const char *name, gemm_function gemm, int size, int runs) {
  double *A = (double *)malloc(size * size * sizeof(double));
  double *B = (double *)malloc(size * size * sizeof(double));
  double *C = (double *)malloc(size * size * sizeof(double));

  if (A == NULL || B == NULL || C == NULL) {
    printf("Memory allocation failed!\n");
    free(A);
    free(B);
    free(C);
    return;
  }

  srand(time(NULL));
  for (int i = 0; i < size * size; i++) {
    A[i] = (double)rand() / RAND_MAX;
    B[i] = (double)rand() / RAND_MAX;
    C[i] = 0.0;
  }

  double total_time = 0.0;
  for (int run = 0; run < runs; run++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    gemm(size, size, size, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    total_time += time_spent;
  }
  double average_time = total_time / runs;

  printf("[%s]\taverage time:\t%.1f ms\n", name, average_time * 1000);

  free(A);
  free(B);
  free(C);
}

int test_gemm(const char *name,
              void (*gemm_func)(int, int, int, double *, double *, double *)) {
  int success = 1;

  // Test case 1
  {
    int m, n, k;
    double *A;
    double *B;

    double *C_expected;

    if (read_matrix("./test_data/A1.txt", &A, &m, &k) == -1 ||
        read_matrix("./test_data/B1.txt", &B, &k, &n) == -1 ||
        read_matrix("./test_data/C1.txt", &C_expected, &m, &n) == -1) {
      printf("Error reading matrices from files.\n");
      free(A);
      free(B);
      free(C_expected);
      return 1;
    }
    double *C = (double *)malloc(m * n * sizeof(double));
    gemm_func(m, n, k, A, B, C);
    if (!matrices_are_equal(m, n, C, C_expected, 1e-12)) {
      success = 0;
    }
  }

  // Test case 2
  {
    int m, n, k;
    double *A;
    double *B;

    double *C_expected;

    if (read_matrix("./test_data/A2.txt", &A, &m, &k) == -1 ||
        read_matrix("./test_data/B2.txt", &B, &k, &n) == -1 ||
        read_matrix("./test_data/C2.txt", &C_expected, &m, &n) == -1) {
      printf("Error reading matrices from files.\n");
      free(A);
      free(B);
      free(C_expected);
      return 1;
    }
    double *C = (double *)malloc(m * n * sizeof(double));
    gemm_func(m, n, k, A, B, C);
    if (!matrices_are_equal(m, n, C, C_expected, 1e-12)) {
      success = 0;
    }
  }

  // Test case 3
  {
    int m, n, k;
    double *A;
    double *B;

    double *C_expected;

    if (read_matrix("./test_data/A3.txt", &A, &m, &k) == -1 ||
        read_matrix("./test_data/B3.txt", &B, &k, &n) == -1 ||
        read_matrix("./test_data/C3.txt", &C_expected, &m, &n) == -1) {
      printf("Error reading matrices from files.\n");
      free(A);
      free(B);
      free(C_expected);
      return 1;
    }
    double *C = (double *)malloc(m * n * sizeof(double));
    gemm_func(m, n, k, A, B, C);
    if (!matrices_are_equal(m, n, C, C_expected, 1e-12)) {
      success = 0;
    }
  }

  if (success) {
    printf("[%s]\ttest:\t\033[1;32mPASS\033[0m\n", name);
  } else {
    printf("[%s]\ttest:\t\033[1;31mFAIL\033[0m\n", name);
  }

  return success;
}

void openblas_gemm(int m, int n, int k, double *A, double *B, double *C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B,
              n, 0.0, C, n);
}

void gemm_iter0(int m, int n, int k, double *A, double *B, double *C) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < k; p++) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

void gemm_iter1(int m, int n, int k, double *A, double *B, double *C) {
  for (int i = 0; i < m; i++) {
    for (int p = 0; p < k; p++) {
      for (int j = 0; j < n; j++) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

void gemm_iter2(int m, int n, int k, double *A, double *B, double *C) {
  for (int i = 0; i < m; i++) {
    for (int p = 0; p < k; p++) {
      __m256d A_val = _mm256_set1_pd(A[i * k + p]);
      for (int j = 0; j < n; j += 4) {
        __m256d B_vec = _mm256_loadu_pd(&B[p * n + j]);
        __m256d C_vec = _mm256_loadu_pd(&C[i * n + j]);
        C_vec = _mm256_fmadd_pd(A_val, B_vec, C_vec);
        _mm256_storeu_pd(&C[i * n + j], C_vec);
      }
      // Handle remaining elements (if any) in a scalar manner
      for (int j = n - (n % 4); j < n; j++) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

void gemm_iter3(int m, int n, int k, double *A, double *B, double *C) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < m; i++) {
    for (int p = 0; p < k; p++) {
      __m256d A_val = _mm256_set1_pd(A[i * k + p]);
      for (int j = 0; j < n; j += 4) {
        __m256d B_vec = _mm256_loadu_pd(&B[p * n + j]);
        __m256d C_vec = _mm256_loadu_pd(&C[i * n + j]);
        C_vec = _mm256_fmadd_pd(A_val, B_vec, C_vec);
        _mm256_storeu_pd(&C[i * n + j], C_vec);
      }
      // Handle remaining elements (if any) in a scalar manner
      for (int j = n - (n % 4); j < n; j++) {
        C[i * n + j] += A[i * k + p] * B[p * n + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_size> <runs>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);
  int runs = atoi(argv[2]);

  if (size <= 0 || runs <= 0) {
    fprintf(stderr,
            "Both matrix size and number of runs must be positive integers.\n");
    return 1;
  }

  assert(test_gemm("openblas_gemm", openblas_gemm));
  assert(test_gemm("gemm_iter0", gemm_iter0));
  assert(test_gemm("gemm_iter1", gemm_iter1));
  assert(test_gemm("gemm_iter2", gemm_iter2));
  assert(test_gemm("gemm_iter3", gemm_iter3));

  benchmark_gemm("openblas_gemm", openblas_gemm, size, runs);
  benchmark_gemm("gemm_iter0", gemm_iter0, size, runs);
  benchmark_gemm("gemm_iter1", gemm_iter1, size, runs);
  benchmark_gemm("gemm_iter2", gemm_iter2, size, runs);
  benchmark_gemm("gemm_iter3", gemm_iter3, size, runs);

  return 0;
}