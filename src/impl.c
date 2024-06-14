#include <immintrin.h>
#include <mm_malloc.h>
#include <omp.h>
#include <string.h>

void do_block(double *p, double *p_next, int i, int j, int N) {
  __m256d double_vec_0 = _mm256_set1_pd(0.25);
  for (int m = 0; m < 4; m++) {
    __m256d line_above = _mm256_loadu_pd(&p[(i + m - 1) * N + j]);
    __m256d line_below = _mm256_loadu_pd(&p[(i + m + 1) * N + j]);

    __m256d line_lleft = _mm256_loadu_pd(&p[(i + m) * N + j - 1]);
    __m256d line_right = _mm256_loadu_pd(&p[(i + m) * N + j + 1]);

    __m256d a = _mm256_add_pd(line_lleft, line_right);
    __m256d b = _mm256_add_pd(line_above, line_below);
    __m256d c = _mm256_mul_pd(_mm256_add_pd(a, b), double_vec_0);
    _mm256_storeu_pd(&p_next[(i + m) * N + j], c);
  }
}

void impl(int N, int step, double *p) {
  if (step % 2 == 1)
    step--;
  double *p_next = (double *)_mm_malloc(N * N * sizeof(double), 64);
  memset(p_next, 0, N * N * sizeof(double));
  memcpy(p_next, p, N * N * sizeof(double));

  int length = N - 1;
  int align = length - length % 4;

  for (int k = 0; k < step; k++) {
#pragma omp parallel for num_threads(4)
    for (int i = 1; i < align + 1; i += 4) {
      for (int j = 1; j < align + 1; j += 4) {
        do_block(p, p_next, i, j, N);
      }
    }
    for (int i = align + 1; i < N - 1; i++) {
      for (int j = 1; j < N - 1; j++) {
        p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
                             p[i * N + j + 1] + p[i * N + j - 1]) /
                            4.0f;
      }
    }
    for (int i = 1; i < N - 1; i++) {
      for (int j = align + 1; j < N - 1; j++) {
        p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
                             p[i * N + j + 1] + p[i * N + j - 1]) /
                            4.0f;
      }
    }
    double *temp = p;
    p = p_next;
    p_next = temp;
  }

  _mm_free(p_next);
  p_next = NULL;
}