#pragma once
#include <cmath>
#include "cuda_check_v1.cuh"
namespace cpde_scalar {
enum {Constant, Leaf, Field, Add, Sub, Mul, Div, Power, Neg, Exp, Log, Sqrt, Sin, Cos, Tanh, Attach};
struct Node { int op,a,b; double parameter; };
__host__ __device__ inline double value(int op,double a,double b,double p) {
    switch(op){
    case Constant:return p;case Add:return a+b;case Sub:return a-b;case Mul:return a*b;
    case Div:return b==0?NAN:a/b;
    case Power:return (a<0&&floor(p)!=p)||(a==0&&p<0)?NAN:pow(a,p);
    case Neg:return -a;case Exp:return exp(a);case Log:return a>0?log(a):NAN;
    case Sqrt:return a>=0?sqrt(a):NAN;case Sin:return sin(a);case Cos:return cos(a);
    case Tanh:return tanh(a);case Attach:return a;default:return NAN;
    }
}
__host__ __device__ inline void partials(int op,double a,double b,double v,double p,double& da,double& db){
    da=db=0;
    switch(op){
    case Add:case Attach:da=db=1;break;case Sub:da=1;db=-1;break;
    case Mul:da=b;db=a;break;case Div:da=1/b;db=-v/b;break;
    case Power:da=p==0?0:(p==1?1:p*pow(a,p-1));break;case Neg:da=-1;break;
    case Exp:da=v;break;case Log:da=1/a;break;case Sqrt:da=.5/v;break;
    case Sin:da=cos(a);break;case Cos:da=-sin(a);break;case Tanh:da=1-v*v;break;
    }
}
}
