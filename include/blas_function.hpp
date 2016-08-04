#ifndef PYCONVNET_BLAS_FUNCTION_HPP
#define PYCONVNET_BLAS_FUNCTION_HPP
extern "C"
{
# include "cblas.h"
}

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
          const int M, const int N, const int K, const float alpha,
          const float* A, const float* B, const float beta, float* C);
void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
          const int M, const int N, const int K, const float alpha,
          const float* A, const float* B, const float beta, float* C) {
    int lda = (TransA == CblasNoTrans) ? K:M;
    int ldb = (TransB == CblasNoTrans) ? N:K;
    int ldc = N;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
                ldb, beta, C, N);
}
#endif //PYCONVNET_BLAS_FUNCTION_HPP
