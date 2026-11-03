#pragma once
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>
#include <unordered_map>
#include <map>
#include <vector>
#include <stdexcept>
#include "scalar_ops.cuh"
#include "reduction_kernels.cuh"
#ifdef CPDE_ENABLE_NVRTC
#include <nvrtc.h>
#include <cuda.h>
#endif

namespace cpde_expression {
using cpde_scalar::Node;
inline std::string number(double x){std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();}
inline std::string forward(const Node& n,const std::string& a,const std::string& b){
    switch(n.op){
    case 0:return number(n.parameter);
    case 1:return "double(y["+std::to_string(n.a)+"/outputs][p*outputs+"+std::to_string(n.a)+"%outputs])";
    case 2:return "fields["+std::to_string(n.a)+"*samples+offset+p]";
    case 3:return a+"+"+b;case 4:return a+"-"+b;case 5:return a+"*"+b;
    case 6:return "("+b+"==0?nan(\"\"):"+a+"/"+b+")";
    case 7:return "pow("+a+","+number(n.parameter)+")";case 8:return "-"+a;
    default:{static const char* names[]={"exp","log","sqrt","sin","cos","tanh"};return std::string(names[n.op-9])+"("+a+")";}
    }
}
inline std::string source(const std::vector<Node>& nodes){
    std::ostringstream s;
    s<<"extern \"C\" __global__ void expression(float** y,float** g,const double* fields,const double* weights,double* loss,int* bad,int outputs,int derivatives,int samples,int width,int valid,int offset,const double* multiplier,int collect,int gradient){\n";
    s<<"for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<width;p+=blockDim.x*gridDim.x){loss[p]=0;bad[p]=-2;if(gradient)for(int d=0;d<derivatives;d++)for(int o=0;o<outputs;o++)g[d][p*outputs+o]=0;if(p>=valid||weights[offset+p]==0)continue;double scale=multiplier?*multiplier:1;if(gradient&&scale==0&&!collect)continue;\n";
    std::vector<bool> active;
    for(size_t i=0;i<nodes.size();++i){
        const auto& n=nodes[i];std::string a="v"+std::to_string(n.a),b="v"+std::to_string(n.b);
        s<<"double v"<<i<<"="<<forward(n,a,b)<<";if(!isfinite(v"<<i<<"))bad[p]=-1;\n";
        bool act=n.op==1?n.b==0:(n.op<=2?false:active[n.a]||(n.op>=3&&n.op<=6&&active[n.b]));active.push_back(act);
    }
    s<<"if(collect)loss[p]=weights[offset+p]*v"<<nodes.size()-1<<";if(!isfinite(loss[p]))bad[p]=-1;if(!gradient||scale==0||bad[p]!=-2)continue;\n";
    for(size_t i=0;i<nodes.size();++i)s<<"double d"<<i<<"="<<(i+1==nodes.size()?"1":"0")<<";\n";
    for(int i=int(nodes.size())-1;i>=0;--i){
        const auto& n=nodes[i];if(n.op<=2||!active[i])continue;
        std::string a="v"+std::to_string(n.a),b="v"+std::to_string(n.b),v="v"+std::to_string(i),da="0",db="0";
        switch(n.op){
        case 3:da=db="1";break;case 4:da="1";db="-1";break;case 5:da=b;db=a;break;
        case 6:da="1/"+b;db="-"+v+"/"+b;break;
        case 7:da=n.parameter==0?"0":(n.parameter==1?"1":number(n.parameter)+"*pow("+a+","+number(n.parameter-1)+")");break;
        case 8:da="-1";break;case 9:da=v;break;case 10:da="1/"+a;break;case 11:da=".5/"+v;break;
        case 12:da="cos("+a+")";break;case 13:da="-sin("+a+")";break;case 14:da="1-"+v+"*"+v;break;
        }
        s<<"if(d"<<i<<"!=0){";
        if(active[n.a])s<<"d"<<n.a<<"+=d"<<i<<"*("<<da<<");";
        if(n.op>=3&&n.op<=6&&active[n.b])s<<"d"<<n.b<<"+=d"<<i<<"*("<<db<<");";
        s<<"}\n";
    }
    // Sum aliases in double before converting to the engine's float matrices.
    std::map<int,std::vector<int>> components;
    for(size_t i=0;i<nodes.size();++i)if(nodes[i].op==1&&nodes[i].b==0)components[nodes[i].a].push_back(i);
    for(const auto& entry:components){
        s<<"{double v=0";for(int i:entry.second)s<<"+d"<<i;
        s<<";float out=float(v*weights[offset+p]*scale);g["<<entry.first<<"/outputs][p*outputs+"<<entry.first<<"%outputs]=out;if(!isfinite(out))bad[p]="<<entry.first<<";}\n";
    }
    s<<"}}\n";return s.str();
}

#if defined(CPDE_HOST_PROBE) || defined(CPDE_CPU_BACKEND)
// CPU scalar-expression interpreter, also used by the independent host probe.
// CUDA builds use specialized NVRTC kernels instead.
inline void reference(const std::vector<Node>& nodes,float** y,float** g,const double* fields,const double* weights,double* loss,int* bad,int outputs,int derivatives,int samples,int width,int valid,int offset,const double* multiplier,bool collect,bool gradient){
    std::vector<double> v(nodes.size()),adj(nodes.size());std::vector<bool> active(nodes.size());
    for(int p=0;p<width;++p){
        loss[p]=0;bad[p]=-2;if(gradient)for(int d=0;d<derivatives;++d)for(int o=0;o<outputs;++o)g[d][p*outputs+o]=0;
        double scale=multiplier?*multiplier:1;
        if(p>=valid||weights[offset+p]==0||(gradient&&scale==0&&!collect))continue;
        std::fill(adj.begin(),adj.end(),0);std::fill(active.begin(),active.end(),false);
        for(size_t i=0;i<nodes.size();++i){auto n=nodes[i];
            if(n.op==1){v[i]=y[n.a/outputs][p*outputs+n.a%outputs];active[i]=!n.b;}
            else if(n.op==2)v[i]=fields[n.a*samples+offset+p];
            else {v[i]=cpde_scalar::value(n.op,n.a<0?0:v[n.a],n.b<0?0:v[n.b],n.parameter);if(n.op>2)active[i]=active[n.a]||(n.op<=6&&active[n.b]);}
            if(!std::isfinite(v[i]))bad[p]=-1;
        }
        if(collect)loss[p]=weights[offset+p]*v.back();if(!std::isfinite(loss[p]))bad[p]=-1;
        if(!gradient||scale==0||bad[p]!=-2)continue;adj.back()=1;
        for(int i=int(nodes.size())-1;i>=0;--i){auto n=nodes[i];if(n.op<=2||!active[i]||adj[i]==0)continue;double a,b;
            cpde_scalar::partials(n.op,v[n.a],n.b<0?0:v[n.b],v[i],n.parameter,a,b);
            if(active[n.a])adj[n.a]+=adj[i]*a;if(n.op<=6&&active[n.b])adj[n.b]+=adj[i]*b;
        }
        std::map<int,double> sums;for(size_t i=0;i<nodes.size();++i)if(nodes[i].op==1&&!nodes[i].b)sums[nodes[i].a]+=adj[i];
        for(auto entry:sums){float v=float(entry.second*weights[offset+p]*scale);g[entry.first/outputs][p*outputs+entry.first%outputs]=v;if(!std::isfinite(v))bad[p]=entry.first;}
    }
}
#endif

struct Program {
    std::vector<Node> nodes;std::string code,log;
#ifdef CPDE_ENABLE_NVRTC
    CUmodule module=nullptr;CUfunction kernel=nullptr;
    static void driver(CUresult r){if(r!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
    ~Program(){if(module)cuModuleUnload(module);}
#endif
    explicit Program(std::vector<Node> ir):nodes(std::move(ir)),code(source(nodes)){}
    void compile(){
#ifdef CPDE_ENABLE_NVRTC
        if(module)return;
        int device=0;checkCudaErrors(cudaGetDevice(&device));cudaDeviceProp prop;checkCudaErrors(cudaGetDeviceProperties(&prop,device));
        std::string arch="--gpu-architecture=compute_"+std::to_string(prop.major)+std::to_string(prop.minor);
        const char* options[]={"--std=c++17",arch.c_str()};nvrtcProgram program;
        nvrtcResult result=nvrtcCreateProgram(&program,code.c_str(),"cpde_expression.cu",0,nullptr,nullptr);
        if(result!=NVRTC_SUCCESS)throw std::runtime_error(nvrtcGetErrorString(result));
        result=nvrtcCompileProgram(program,2,options);size_t size=0;nvrtcGetProgramLogSize(program,&size);log.resize(size);if(size)nvrtcGetProgramLog(program,&log[0]);
        if(result!=NVRTC_SUCCESS){nvrtcDestroyProgram(&program);throw std::runtime_error("expression compile failed: "+log);}
        nvrtcGetPTXSize(program,&size);std::vector<char> ptx(size);nvrtcGetPTX(program,ptx.data());nvrtcDestroyProgram(&program);
        driver(cuModuleLoadData(&module,ptx.data()));driver(cuModuleGetFunction(&kernel,module,"expression"));
#elif defined(CPDE_CPU_BACKEND)
        log="CPU backend: scalar IR interpreter; displayed CUDA source is a reference, not executed code.";
#elif !defined(CPDE_HOST_PROBE)
        throw std::runtime_error("this executable was built without NVRTC expression support");
#endif
    }
    void launch(float** y,float** g,const double* fields,const double* weights,double* loss,int* bad,int outputs,int derivatives,int samples,int width,int valid,int offset,const double* multiplier,int collect,int gradient,cudaStream_t stream){
#ifdef CPDE_ENABLE_NVRTC
        void* args[]={&y,&g,&fields,&weights,&loss,&bad,&outputs,&derivatives,&samples,&width,&valid,&offset,&multiplier,&collect,&gradient};
        driver(cuLaunchKernel(kernel,cpde_reduce::blocks_for(width),1,1,cpde_reduce::threads,1,1,0,stream,args,nullptr));
#elif defined(CPDE_HOST_PROBE) || defined(CPDE_CPU_BACKEND)
        reference(nodes,y,g,fields,weights,loss,bad,outputs,derivatives,samples,width,valid,offset,multiplier,collect,gradient);
#else
        throw std::runtime_error("NVRTC expression support unavailable");
#endif
    }
};
}
