#ifndef PYCONVNET_BLAS_FUNCTION_HPP
#define PYCONVNET_BLAS_FUNCTION_HPP
extern "C"
{
# include "cblas.h"
}

void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
          const int M, const int N, const int K, const float alpha,
          const float* A, const float* B, const float beta, float* C);

#endif //PYCONVNET_BLAS_FUNCTION_HPP
