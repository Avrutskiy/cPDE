#pragma once
#include "cuda_runtime.h"
using cublasHandle_t=int;
enum cublasStatus_t{CUBLAS_STATUS_SUCCESS,CUBLAS_STATUS_NOT_INITIALIZED,CUBLAS_STATUS_ALLOC_FAILED,CUBLAS_STATUS_INVALID_VALUE,CUBLAS_STATUS_ARCH_MISMATCH,CUBLAS_STATUS_MAPPING_ERROR,CUBLAS_STATUS_EXECUTION_FAILED,CUBLAS_STATUS_INTERNAL_ERROR};
enum{CUBLAS_OP_N,CUBLAS_OP_T};
inline cublasStatus_t cublasCreate(int* h){*h=0;return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasDestroy(int){return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSetStream(int,int){return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSgemm(int,int ta,int tb,int m,int n,int k,const float* alpha,const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc){
 for(int j=0;j<n;++j)for(int i=0;i<m;++i){float sum=0;for(int q=0;q<k;++q)sum+=(ta==CUBLAS_OP_N?a[q*lda+i]:a[i*lda+q])*(tb==CUBLAS_OP_N?b[j*ldb+q]:b[q*ldb+j]);c[j*ldc+i]=*alpha*sum+(*beta==0?0:*beta*c[j*ldc+i]);}return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSger(int,int m,int n,const float* alpha,const float* x,int ix,const float* y,int iy,float* a,int lda){for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[j*lda+i]+=*alpha*x[i*ix]*y[j*iy];return CUBLAS_STATUS_SUCCESS;}
inline cublasStatus_t cublasSgemv(int,int op,int m,int n,const float* alpha,const float* a,int lda,const float* x,int ix,const float* beta,float* y,int iy){
 if(op!=CUBLAS_OP_N)std::abort();for(int i=0;i<m;++i){float sum=0;for(int j=0;j<n;++j)sum+=a[j*lda+i]*x[j*ix];y[i*iy]=*alpha*sum+(*beta==0?0:*beta*y[i*iy]);}return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSasum(int,int n,const float* x,int stride,float* out){float s=0;for(int i=0;i<n;++i)s+=std::fabs(x[i*stride]);*out=s;return CUBLAS_STATUS_SUCCESS;}
