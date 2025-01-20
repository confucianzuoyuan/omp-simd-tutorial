#include <immintrin.h>
#include <mm_malloc.h>
#include <omp.h>
#include <string.h>

void impl(int N, int step, double *p)
{
  double divisor[4] = {
      0.25f,
      0.25f,
      0.25f,
      0.25f,
  };
  __m256d p_divisor = _mm256_loadu_pd(divisor);
  // rearrange
  int N2 = (N + 1) / 2;
  double *p_part[2] = {
      aligned_alloc(32, N2 * N * sizeof(double)),
      aligned_alloc(32, N2 * N * sizeof(double)),
  };
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    int part = i & 1;
    for (int j = 0; j < N; ++j)
    {
      p_part[part][i * N2 + j / 2] = p[i * N + j];
      part ^= 1;
    }
  }
  // caculate
  int INPUTpartID = 1;
  int OUTPUTpartID = 0;
  if (N & 1)
  { // N = odd
    for (int k = 0; k < step; k++)
    {
#pragma omp parallel for
      for (int i = 1; i < N - 1; i++)
      {
        int j_head = (INPUTpartID + i) & 1;
        int j_begin = i * N2 + j_head;
        int j_end = (i + 1) * N2 - 1;
        int j = j_begin;
        for (; j < j_end - 3; j += 4)
        {
          __m256d p1 = _mm256_loadu_pd(&p_part[INPUTpartID][j - N2]);
          __m256d p2 = _mm256_loadu_pd(&p_part[INPUTpartID][j - j_head]);
          __m256d p3 = _mm256_loadu_pd(&p_part[INPUTpartID][1 + j - j_head]);
          __m256d p4 = _mm256_loadu_pd(&p_part[INPUTpartID][j + N2]);
          __m256d sum1 = _mm256_add_pd(p1, p2);
          __m256d sum2 = _mm256_add_pd(p3, p4);
          __m256d sum3 = _mm256_add_pd(sum1, sum2);
          __m256d result = _mm256_mul_pd(sum3, p_divisor);
          _mm256_storeu_pd(&p_part[OUTPUTpartID][j], result);
        }

        // for the tail
        for (; j < j_end; j++)
        {
          double p1 = p_part[INPUTpartID][j - N2];
          double p2 = p_part[INPUTpartID][j - j_head];
          double p3 = p_part[INPUTpartID][1 + j - j_head];
          double p4 = p_part[INPUTpartID][j + N2];
          p_part[OUTPUTpartID][j] = (p1 + p2 + p3 + p4) / 4.0f;
        }
      }
      int temp = INPUTpartID;
      INPUTpartID = OUTPUTpartID;
      OUTPUTpartID = temp;
    }
  }
  else
  { // N = even
    for (int k = 0; k < step; k++)
    {
#pragma omp parallel for
      for (int i = 1; i < N - 1; i++)
      {
        int j_head = (INPUTpartID + i) & 1;
        int j_begin = i * N2 + j_head;
        int j_end = N2 - 1 + j_begin;
        int j = j_begin;
        for (; j < j_end - 3; j += 4)
        {
          __m256d p1 = _mm256_loadu_pd(&p_part[INPUTpartID][j - N2]);
          __m256d p2 = _mm256_loadu_pd(&p_part[INPUTpartID][j - j_head]);
          __m256d p3 = _mm256_loadu_pd(&p_part[INPUTpartID][1 + j - j_head]);
          __m256d p4 = _mm256_loadu_pd(&p_part[INPUTpartID][j + N2]);
          __m256d sum1 = _mm256_add_pd(p1, p2);
          __m256d sum2 = _mm256_add_pd(p3, p4);
          __m256d sum3 = _mm256_add_pd(sum1, sum2);
          __m256d result = _mm256_mul_pd(sum3, p_divisor);
          _mm256_storeu_pd(&p_part[OUTPUTpartID][j], result);
        }

        // for the tail
        for (; j < j_end; j++)
        {
          double p1 = p_part[INPUTpartID][j - N2];
          double p2 = p_part[INPUTpartID][j - j_head];
          double p3 = p_part[INPUTpartID][1 + j - j_head];
          double p4 = p_part[INPUTpartID][j + N2];
          p_part[OUTPUTpartID][j] = (p1 + p2 + p3 + p4) / 4.0f;
        }
      }

      int temp = INPUTpartID;
      INPUTpartID = OUTPUTpartID;
      OUTPUTpartID = temp;
    }
  }
// rearrange back
#pragma omp parallel for
  for (int i = 0; i < N; ++i)
  {
    int part = i & 1;
    for (int j = 0; j < N; ++j)
    {
      p[i * N + j] = p_part[part][i * N2 + j / 2];
      part ^= 1;
    }
  }
  free(p_part[0]);
  free(p_part[1]);
}
