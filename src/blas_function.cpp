# include <cmath>
#include <algorithm>

# include "blas_function.hpp"

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
          const int M, const int N, const int K, const float alpha,
          const float* A, const float* B, const float beta, float* C) {
    int lda = (TransA == CblasNoTrans) ? K:M;
    int ldb = (TransB == CblasNoTrans) ? N:K;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                ldb, beta, C, N);
}

void vector_add(const float* A, const float* B, float* C, const float coeff_a,
                const float coeff_b, const int vector_size) {
    for(int i = 0; i < vector_size; ++i) {
          C[i] = A[i] * coeff_a + B[i] * coeff_b;
    }
}

void vector_sub(const float* A, const float* B, float* C, const int vector_size) {
    for(int i = 0; i < vector_size; ++i) {
        C[i] = A[i] - B[i];
    }
}

void vector_mul(const float* A, const float* B, float* C, const int vector_size) {
    for(int i = 0; i < vector_size; ++i) {
        C[i] = A[i] * B[i];
    }
}

void vector_exp(const float* A, float* B, const int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
        B[i] = expf(A[i]);
    }
}

void vector_sub_scalar(float* A, float b, float* B, const int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
        B[i] = A[i] - b;
    }
}

void vector_div_scalar(float* A, float b, const int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
        A[i] = A[i] / b;
    }
}

void vector_mul_scalar(float* A, float b, const int vector_size) {
    for (int i = 0; i < vector_size; ++i) {
        A[i] = A[i] * b;
    }
}

float vector_sum(float* A, const int vector_size) {
    float sum = 0;
    for (int i = 0; i < vector_size; ++i) {
        sum += A[i];
    }
    return sum;
}

void vector_scale(float* A, const int vector_size) {
    float max_item = *(std::max_element(A, A + vector_size));
    float min_item = *(std::min_element(A, A + vector_size));
    if (fabsf(max_item) < fabsf(min_item))
        max_item = fabsf(min_item);
    if (max_item < 0.1)
        return;
    for (int i = 0; i < vector_size; ++i) {
        A[i] = A[i] / max_item;
    }
}
