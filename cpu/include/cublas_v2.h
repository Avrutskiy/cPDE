#pragma once
// Column-major CPU BLAS adapter for the shared engine. System CBLAS is optional;
// the portable fallback implements exactly the operations used by cPDE.
#include "cuda_runtime.h"
#include <algorithm>
#ifdef CPDE_CPU_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif defined(CPDE_CPU_CBLAS)
#include <cblas.h>
#endif
using cublasHandle_t = int;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED,
    CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH,
    CUBLAS_STATUS_MAPPING_ERROR, CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR };
enum { CUBLAS_OP_N, CUBLAS_OP_T };
inline bool cpu_blas_operation(int op) { return op==CUBLAS_OP_N || op==CUBLAS_OP_T; }
inline cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if(!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle=0; return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t,cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSgemm(cublasHandle_t, int ta, int tb, int m, int n, int k,
    const float* alpha, const float* a, int lda, const float* b, int ldb,
    const float* beta, float* c, int ldc) {
    if(!cpu_blas_operation(ta) || !cpu_blas_operation(tb) || m<0 || n<0 || k<0 ||
       lda<std::max(1,ta==CUBLAS_OP_N?m:k) || ldb<std::max(1,tb==CUBLAS_OP_N?k:n) ||
       ldc<std::max(1,m) || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    if(!m || !n) return CUBLAS_STATUS_SUCCESS;
    if(!c || (k && *alpha!=0 && (!a || !b))) return CUBLAS_STATUS_INVALID_VALUE;
#if defined(CPDE_CPU_ACCELERATE) || defined(CPDE_CPU_CBLAS)
    cblas_sgemm(CblasColMajor,ta==CUBLAS_OP_N?CblasNoTrans:CblasTrans,
                tb==CUBLAS_OP_N?CblasNoTrans:CblasTrans,m,n,k,*alpha,a,lda,b,ldb,*beta,c,ldc);
#else
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
        float sum=0;
        if(*alpha!=0) for(int q=0;q<k;++q)
            sum+=(ta==CUBLAS_OP_N?a[q*lda+i]:a[i*lda+q])*(tb==CUBLAS_OP_N?b[j*ldb+q]:b[q*ldb+j]);
        c[j*ldc+i]=*alpha*sum+(*beta==0?0:*beta*c[j*ldc+i]);
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSger(cublasHandle_t, int m, int n, const float* alpha,
    const float* x, int ix, const float* y, int iy, float* a, int lda) {
    if(m<0 || n<0 || ix<=0 || iy<=0 || lda<std::max(1,m) || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
    if(!m || !n || *alpha==0) return CUBLAS_STATUS_SUCCESS;
    if(!x || !y || !a) return CUBLAS_STATUS_INVALID_VALUE;
#if defined(CPDE_CPU_ACCELERATE) || defined(CPDE_CPU_CBLAS)
    cblas_sger(CblasColMajor,m,n,*alpha,x,ix,y,iy,a,lda);
#else
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) a[j*lda+i]+=*alpha*x[i*ix]*y[j*iy];
#endif
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemv(cublasHandle_t, int op, int m, int n, const float* alpha,
    const float* a, int lda, const float* x, int ix, const float* beta, float* y, int iy) {
    if(!cpu_blas_operation(op) || m<0 || n<0 || lda<std::max(1,m) || ix<=0 || iy<=0 ||
       !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
    if(!m || !n) return CUBLAS_STATUS_SUCCESS;
    if(!y || (*alpha!=0 && (!a || !x))) return CUBLAS_STATUS_INVALID_VALUE;
#if defined(CPDE_CPU_ACCELERATE) || defined(CPDE_CPU_CBLAS)
    cblas_sgemv(CblasColMajor,op==CUBLAS_OP_N?CblasNoTrans:CblasTrans,m,n,*alpha,a,lda,x,ix,*beta,y,iy);
#else
    const int rows=op==CUBLAS_OP_N?m:n, cols=op==CUBLAS_OP_N?n:m;
    for(int i=0;i<rows;++i) {
        float sum=0;
        if(*alpha!=0) for(int j=0;j<cols;++j) sum+=(op==CUBLAS_OP_N?a[j*lda+i]:a[i*lda+j])*x[j*ix];
        y[i*iy]=*alpha*sum+(*beta==0?0:*beta*y[i*iy]);
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSasum(cublasHandle_t,int n,const float* x,int stride,float* out) {
    if(!out || stride<=0 || (n>0 && !x)) return CUBLAS_STATUS_INVALID_VALUE;
    if(n<=0) {*out=0;return CUBLAS_STATUS_SUCCESS;}
#if defined(CPDE_CPU_ACCELERATE) || defined(CPDE_CPU_CBLAS)
    *out=cblas_sasum(n,x,stride);
#else
    float sum=0; for(int i=0;i<n;++i) sum+=std::fabs(x[i*stride]); *out=sum;
#endif
    return CUBLAS_STATUS_SUCCESS;
}
