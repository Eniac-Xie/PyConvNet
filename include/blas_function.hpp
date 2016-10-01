#ifndef PYCONVNET_BLAS_FUNCTION_HPP
#define PYCONVNET_BLAS_FUNCTION_HPP

extern "C"
{
# include "cblas.h"
}

// matrix multiplication
void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
          const int M, const int N, const int K, const float alpha,
          const float* A, const float* B, const float beta, float* C);

// vector operation
void vector_add(const float* A, const float* B, float* C, const float coeff_a,
                const float coeff_b, const int vector_size);

void vector_sub(const float* A, const float* B, float* C, const int vector_size);

void vector_mul(const float* A, const float* B, float* C, const int vector_size);

void vector_exp(const float* A, float* B, const int vector_size);

void vector_sub_scalar(float* A, float b, float* B, const int vector_size);

void vector_div_scalar(float* A, float b, const int vector_size);

void vector_mul_scalar(float* A, float b, const int vector_size);

float vector_sum(float* A, const int vector_size);

void vector_mul_scalar(float* A, float b, const int vector_size);

void vector_scale(float* A, const int vector_size);

#endif //PYCONVNET_BLAS_FUNCTION_HPP
