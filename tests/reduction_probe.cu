// Build with nvcc for real CUB/block/padding checks; the Python suite also runs a
// serial host-probe variant, which does NOT validate CUDA or block collectives.
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <vector>
// Exercise inclusion after the legacy derivative engine's macro definition.
#define E 2.71828182f
#include "cost_kernels.cuh"
static_assert(E > 2.7f && E < 2.8f, "CUB includes must preserve the legacy E macro");

static void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { checkCudaErrors(cudaMalloc((void**)&p,n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    void upload(const std::vector<T>& a) {
        checkCudaErrors(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
};

int main() {
    try {
        cudaStream_t stream;checkCudaErrors(cudaStreamCreate(&stream));
        {
            cpde_reduce::Workspace work;work.reserve(3,2,stream);work.begin(3,stream);
            std::vector<cpde_reduce::CostStats> partials(7,cpde_reduce::empty_cost());
            const double input[]={1e8,1,-1e8,-9,2,0,-1};
            for(int i=0;i<7;++i)partials[i].value=input[i];
            checkCudaErrors(cudaMemcpy(work.cost_blocks,partials.data(),partials.size()*sizeof(partials[0]),cudaMemcpyHostToDevice));
            work.finish_chunk(7,0,stream);
            work.download(3,0,stream);
            expect(work.host_costs[0].value==-7,"signed reduction or double accumulation failed");
            // Reuse scratch for a second chunk and independent cost slot.
            partials.assign(2,cpde_reduce::empty_cost());partials[0].value=2;partials[1].value=3;
            partials[1].bad_sample=12;partials[1].bad_component=7;
            checkCudaErrors(cudaMemcpy(work.cost_blocks,partials.data(),partials.size()*sizeof(partials[0]),cudaMemcpyHostToDevice));
            work.finish_chunk(2,0,stream);work.finish_chunk(2,1,stream);
            work.download(3,0,stream);
            expect(work.host_costs[0].value==-2&&work.host_costs[1].value==5,"chunk/cost accumulation failed");
            expect(work.host_costs[0].bad_sample==12&&work.host_costs[0].bad_component==7,"invalid-value location lost");

            std::vector<float> gradient(4101,0);gradient[0]=3;gradient[300]=-4;gradient[4100]=12;
            Device<float> dg(gradient.size());dg.upload(gradient);
            work.inspect_gradient(dg.p,gradient.size(),0,stream);
            work.download(3,1,stream);
            expect(work.host_gradients[0].squared_norm==169&&work.host_gradients[0].max_abs==12,
                   "gradient norm/max failed across blocks and tail");
            expect(work.host_gradients[0].nonfinite_count==0,"finite gradient marked invalid");
            gradient[1025]=std::numeric_limits<float>::infinity();
            gradient[2050]=std::numeric_limits<float>::quiet_NaN();dg.upload(gradient);
            work.inspect_gradient(dg.p,gradient.size(),1,stream);work.download(3,2,stream);
            expect(work.host_gradients[1].nonfinite_count==2&&work.host_gradients[1].first_nonfinite==1025,
                   "nonfinite count/index failed");

            const int width=777,valid=773;
            std::vector<float> y(width,0),target(valid,.25f),g(width,99);
            double expected=0;
            for(int i=0;i<valid;++i){
                y[i]=float(i%7)-3;
                float residual=y[i]-.25f;expected+=double(1.25f*residual*residual);
            }
            // Invalid padding must be ignored while its dE/dY is cleared.
            for(int i=valid;i<width;++i)y[i]=std::numeric_limits<float>::quiet_NaN();
            Device<float> dy(width),dt(valid),dgout(width),dw(1);
            dy.upload(y);dt.upload(target);dgout.upload(g);dw.upload({1.25f});
            Device<float*> py(1),pt(1),pg(1);py.upload({dy.p});pt.upload({dt.p});pg.upload({dgout.p});
            Device<int> rows(1);rows.upload({0});
            int blocks=cpde_reduce::blocks_for(width);
            work.begin(3,stream);
            CPDE_LAUNCH(matching_cost_kernel<true>, blocks, cpde_reduce::threads, 0, stream, py.p,pg.p,pt.p,
                rows.p,dw.p,work.cost_blocks,1,1,width,valid,0,1);
            checkCudaErrors(cudaGetLastError());work.finish_chunk(blocks,0,stream);work.download(3,0,stream);
            expect(work.host_costs[0].value==expected,"fused cost reduction failed");
            expect(work.host_costs[0].bad_sample==INT_MAX,"padding was evaluated");
            checkCudaErrors(cudaMemcpy(g.data(),dgout.p,width*sizeof(float),cudaMemcpyDeviceToHost));
            for(int i=0;i<width;++i)expect(g[i]==(i<valid?2.5f*(y[i]-.25f):0),"output gradient/padding mismatch");

            work.begin(3,stream);
            CPDE_LAUNCH(matching_cost_kernel<false>, blocks, cpde_reduce::threads, 0, stream, py.p,pg.p,pt.p,
                rows.p,dw.p,work.cost_blocks,1,1,width,valid,0,1);
            checkCudaErrors(cudaGetLastError());work.finish_chunk(blocks,0,stream);work.download(3,0,stream);
            expect(work.host_costs[0].value==0,"unreported loss was accumulated");
            expect(work.host_costs[0].bad_sample==INT_MAX,"validation changed without loss collection");
        }
        checkCudaErrors(cudaStreamDestroy(stream));
        std::cout<<"signed sums, scratch reuse, gradient statistics, fused costs, and padding passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
