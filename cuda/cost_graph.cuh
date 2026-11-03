#pragma once
#include "scalar_ops.cuh"
#include "reduction_kernels.cuh"
namespace cpde_graph {
struct Summary {double value;int bad_node;};
__global__ void execute(const cpde_scalar::Node* nodes,int count,int root,const cpde_reduce::CostStats* integrals,int leaves,
    double* values,double* adjoints,double* multipliers,int* active,Summary* summary,bool gradient){
    if(blockIdx.x||threadIdx.x)return;
    summary->bad_node=-1;summary->value=0;
    for(int i=0;i<leaves;++i)multipliers[i]=0;
    for(int i=0;i<count;++i){auto n=nodes[i];adjoints[i]=0;active[i]=n.op==1;
        if(n.op==1)values[i]=integrals[n.a].value;
        else {values[i]=cpde_scalar::value(n.op,n.a<0?0:values[n.a],n.b<0?0:values[n.b],n.parameter);
            if(n.op>2)active[i]=active[n.a]||((n.op<=6||n.op==15)&&active[n.b]);}
        if(!isfinite(values[i])&&summary->bad_node<0)summary->bad_node=i;
    }
    summary->value=values[root];if(!gradient||summary->bad_node>=0)return;adjoints[root]=1;
    for(int i=count-1;i>=0;--i){auto n=nodes[i];double seed=adjoints[i];if(seed==0||!active[i])continue;
        if(n.op==1)multipliers[n.a]+=seed;
        else if(n.op>2){double a,b;cpde_scalar::partials(n.op,values[n.a],n.b<0?0:values[n.b],values[i],n.parameter,a,b);
            if(active[n.a])adjoints[n.a]+=seed*a;
            if((n.op<=6||n.op==15)&&active[n.b])adjoints[n.b]+=seed*b;
        }
        if(!isfinite(seed)&&summary->bad_node<0)summary->bad_node=i;
    }
    for(int i=0;i<leaves;++i)if(!isfinite(multipliers[i]))summary->bad_node=root;
}

__global__ void scale_gradients(float** gradients,int outputs,int derivatives,int width,const double* multiplier){
    for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<width;p+=blockDim.x*gridDim.x)
        for(int d=0;d<derivatives;++d)for(int o=0;o<outputs;++o){
            float& v=gradients[d][p*outputs+o];v=*multiplier==0?0:float(v*(*multiplier));
        }
}
__global__ void sample_blocks(const double* values,const int* bad,int width,int valid,int offset,cpde_reduce::CostStats* partials){
    auto local=cpde_reduce::empty_cost();
    for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<valid;p+=blockDim.x*gridDim.x){
        local.value+=values[p];if(bad[p]!=-2)cpde_reduce::mark_bad(local,offset+p,bad[p]);
    }
    using Reduce=cub::BlockReduce<cpde_reduce::CostStats,cpde_reduce::threads>;
    __shared__ typename Reduce::TempStorage storage;auto result=Reduce(storage).Reduce(local,cpde_reduce::CostCombine{});
    if(threadIdx.x==0)partials[blockIdx.x]=result;
}
}
