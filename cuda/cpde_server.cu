#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <numeric>
#include <climits>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include "protocol.hpp"
#include "cpde_engine.cuh"
#include "cost_kernels.cuh"
#include "expression_program.hpp"
#include "cost_graph.cuh"
#include "graph_schedule.hpp"

using wire::Buffer;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static int checked_size(int a,int b,int c=1){
    int64_t n=int64_t(a)*b*c;require(a>0&&b>0&&c>0&&n<=INT_MAX,"invalid or excessive dimensions");return int(n);
}
template<class T> struct DeviceArray {
    T* ptr=nullptr; size_t size=0;
    ~DeviceArray(){if(ptr)cudaFree(ptr);}
    void upload(const std::vector<T>& a){
        if(ptr)checkCudaErrors(cudaFree(ptr));ptr=nullptr;size=a.size();
        if(size){checkCudaErrors(cudaMalloc((void**)&ptr,size*sizeof(T)));checkCudaErrors(cudaMemcpy(ptr,a.data(),size*sizeof(T),cudaMemcpyHostToDevice));}
    }
};
struct DataRecord {
    std::unique_ptr<inout> gpu;
    uint64_t version=0;
};
struct GraphRecord {
    DeviceArray<cpde_scalar::Node> nodes;
    DeviceArray<double> values,adjoints,multipliers;
    DeviceArray<double> fixed_multipliers;
    std::vector<cpde_graph::KnownScalar> schedule;
    DeviceArray<int> active;
    DeviceArray<cpde_graph::Summary> summary;
    cpde_graph::Summary host{0,-1};int root=0;
    void read(Buffer& b,int leaves){
        auto raw=b.array<int>();auto constants=b.array<double>();root=b.get<int>();
        require(!constants.empty()&&constants.size()<=4096&&raw.size()==3*constants.size()&&root>=0&&root<int(constants.size()),"invalid cost graph size");
        std::vector<cpde_scalar::Node> ir;
        for(int i=0;i<int(constants.size());++i){int op=raw[3*i],a=raw[3*i+1],c=raw[3*i+2];
            require(std::isfinite(constants[i]),"nonfinite graph constant");
            require(op==0||op==1||(op>=3&&op<=15),"invalid graph opcode");
            if(op==1)require(a>=0&&a<leaves,"invalid graph leaf");
            if(op>=3){require(a>=0&&a<i,"graph must be topologically ordered");if(op<=6||op==15)require(c>=0&&c<i,"invalid graph child");}
            ir.push_back({op,a,c,constants[i]});
        }
        nodes.upload(ir);values.upload(std::vector<double>(ir.size()));adjoints.upload(std::vector<double>(ir.size()));
        active.upload(std::vector<int>(ir.size()));multipliers.upload(std::vector<double>(leaves));summary.upload({host});
        schedule=cpde_graph::schedule(ir,root,leaves);
        std::vector<double> fixed(leaves,0);
        for(int i=0;i<leaves;++i)if(schedule[i].known)fixed[i]=schedule[i].value;
        fixed_multipliers.upload(fixed);
    }
    void run(const cpde_reduce::CostStats* costs,int leaves,bool gradient,cudaStream_t stream){
        CPDE_LAUNCH(cpde_graph::execute, 1, 1, 0, stream, nodes.ptr,nodes.size,root,costs,leaves,values.ptr,adjoints.ptr,multipliers.ptr,active.ptr,summary.ptr,gradient);
        checkCudaErrors(cudaGetLastError());
    }
    void queue_download(cudaStream_t stream){checkCudaErrors(cudaMemcpyAsync(&host,summary.ptr,sizeof(host),cudaMemcpyDeviceToHost,stream));}
    void validate(){if(host.bad_node>=0)throw std::runtime_error("invalid value or derivative in cost graph node "+std::to_string(host.bad_node));}
};
struct NetRecord {
    std::unique_ptr<network> gpu;
    cpde_reduce::Workspace reductions;
    std::vector<int> neurons,linear;
    int capacity=2,optimizer=-1;
    uint64_t updates=0,adam_step=0,stage_done=0,remaining=0;
    float learning_rate=.001f,initial_step=.002f;
    std::string stage;
    uint64_t loaded_id=0,loaded_version=0;
    int loaded_width=0,loaded_offset=0,loaded_valid=0;
    std::vector<int> loaded_rows;
};
struct CostRecord {
    uint64_t network_id,input_id,target_id;
    int kind,width,equations=0,weight_samples=0;
    std::vector<int> rows;
    std::unique_ptr<itable> schema;
    DeviceArray<int> target_rows;
    DeviceArray<float> weights,coefficients,rhs;
    DeviceArray<PolynomialTerm> terms;
    std::shared_ptr<cpde_expression::Program> program;
    DeviceArray<double> fields,quadrature_weights,sample_values;
    DeviceArray<int> sample_errors;
    int samples=0;
    bool has_components=true,has_gradient=true;
};
struct Server {
    int device; uint64_t next_id=1;
    std::unordered_map<uint64_t,std::unique_ptr<DataRecord>> data;
    std::unordered_map<uint64_t,std::unique_ptr<NetRecord>> nets;
    std::unordered_map<uint64_t,std::unique_ptr<CostRecord>> costs;
    std::unordered_map<std::string,std::shared_ptr<cpde_expression::Program>> programs;
    explicit Server(int d):device(d){checkCudaErrors(cudaSetDevice(device));}
    template<class M> static typename M::mapped_type::element_type& lookup(M& m,uint64_t id){
        auto p=m.find(id);require(p!=m.end(),"unknown or released object handle");return *p->second;
    }
    DataRecord& dat(uint64_t id){return lookup(data,id);}
    NetRecord& net(uint64_t id){return lookup(nets,id);}
    static std::vector<int> table(inout& d){
        std::vector<int> a(d.S->h*d.S->w);
        for(int r=0;r<d.S->h;++r)for(int c=0;c<d.S->w;++c)a[r*d.S->w+c]=d.S->host[c*d.S->h+r];return a;
    }
    static std::unique_ptr<itable> make_schema(inout& d,const std::vector<int>& rows){
        require(!rows.empty(),"empty derivative closure");
        std::unique_ptr<itable> s(new itable(rows.size(),d.S->w));
        int previous=-1;
        for(size_t r=0;r<rows.size();++r){
            int src=rows[r];require(src>=0&&src<d.S->h,"invalid derivative row");
            int order=d.S->host[src];require(order>=previous&&order<=6,"derivative closure must be sorted, orders 0..6");previous=order;
            for(int c=0;c<s->w;++c)s->host[c*s->h+r]=d.S->host[c*d.S->h+src];
        }
        require(s->host[0]==0,"value derivative must come first");s->todev();return s;
    }
    static std::vector<int> all_rows(inout& d){
        std::vector<int> rows(d.S->h);std::iota(rows.begin(),rows.end(),0);
        std::stable_sort(rows.begin(),rows.end(),[&](int a,int b){return d.S->host[a]<d.S->host[b];});return rows;
    }
    void reserve(NetRecord& n,int need,int scratch,int width){
        network& m=*n.gpu;
        int maximum=*std::max_element(n.neurons.begin(),n.neurons.end());
        int cache_size=checked_size(scratch,maximum);
        if(need<=n.capacity&&cache_size<=m.cache->mem->size&&width<=m.ones->w)return;
        checkCudaErrors(cudaStreamSynchronize(*m.str->exec));
        if(need>n.capacity){
            for(layer* l=m.layers;l;l=l->next){
                checked_size(need,l->n);l->wipe_context();delete l->mem;l->mem=new memory(need*l->n,'d');
            }
            n.capacity=need;n.loaded_id=0;
        }
        if(cache_size>m.cache->mem->size){
            m.cache->wipe_context();delete m.cache->mem;m.cache->mem=new memory(cache_size,'d');
            m.cache->n=maximum;
        }
        if(width>m.ones->w){
            delete m.ones;m.ones=new matrix(1,width,'d');m.ones->init(1.0f,'v',*m.str->exec);
        }
    }
    void prepare(NetRecord& n,itable* schema,int width,bool backward=true){
        int scratch=checked_size(width,schema->h);
        reserve(n,checked_size(scratch,backward?2:1),scratch,width);
        std::shared_ptr<derivative_tables> tables;
        for(layer* l=n.gpu->layers;l;l=l->next){
            if(backward&&l->lin==0&&!tables)tables=n.gpu->shared_tables(schema);
            l->add_context(schema,l->n,width,backward?1:0,tables);
            if(l->lin==0)n.gpu->cache->add_context(schema,l->n,width,0);
        }
    }
    void activate(NetRecord& n,uint64_t data_id,itable* schema,const std::vector<int>& rows,int width,int offset,int valid,bool backward=true){
        auto& record=dat(data_id);auto& input=*record.gpu;
        network& m=*n.gpu;
        m.layers->env=m.layers->get_context(schema,input.h,width,backward?1:0);
        require(m.layers->env!=nullptr,"missing prepared context");require(m.prop_context()==0,"context propagation failed");
        if(n.loaded_id==data_id&&n.loaded_version==record.version&&n.loaded_width==width&&n.loaded_offset==offset&&n.loaded_valid==valid&&n.loaded_rows==rows)return;
        n.loaded_id=data_id;n.loaded_version=record.version;n.loaded_width=width;n.loaded_offset=offset;n.loaded_valid=valid;n.loaded_rows=rows;
        for(size_t d=0;d<rows.size();++d){
            float* dest=m.layers->env->ADDR->host[d];
            checkCudaErrors(cudaMemcpyAsync(dest,input.ADDR->host[rows[d]]+offset*input.h,valid*input.h*sizeof(float),cudaMemcpyDeviceToDevice,*m.str->exec));
            if(valid<width)checkCudaErrors(cudaMemsetAsync(dest+valid*input.h,0,(width-valid)*input.h*sizeof(float),*m.str->exec));
        }
    }
    uint64_t evaluate(uint64_t ni,uint64_t di,int chunk){
        auto& n=net(ni);auto& input=*dat(di).gpu;require(input.h==n.neurons.front(),"input channel mismatch");
        int width=chunk?std::min(chunk,input.w):input.w;require(width>0,"invalid chunk size");
        auto rows=all_rows(input);auto s=make_schema(input,rows);prepare(n,s.get(),width,false);
        auto out=std::unique_ptr<DataRecord>(new DataRecord);
        out->gpu.reset(new inout(n.neurons.back(),input.w,input.S,device));
        for(int offset=0;offset<input.w;offset+=width){
            int valid=std::min(width,input.w-offset);activate(n,di,s.get(),rows,width,offset,valid,false);
            layer* l=n.gpu->feedforward();
            for(size_t d=0;d<rows.size();++d)checkCudaErrors(cudaMemcpyAsync(out->gpu->ADDR->host[rows[d]]+offset*l->n,l->env->ADDR->host[d],valid*l->n*sizeof(float),cudaMemcpyDeviceToDevice,*n.gpu->str->exec));
        }
        checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));uint64_t id=next_id++;data[id]=std::move(out);return id;
    }
    void reset(NetRecord& n,int method){
        checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));
        for(layer* l=n.gpu->layers;l->W_plus;l=l->next){
            weights& w=*l->W_plus;w.act_grad=1;
            for(int b=1;b<4;++b)checkCudaErrors(cudaMemsetAsync(w.W[b]->mem->vals,0,w.Wh*(w.Ww+1)*sizeof(float),*n.gpu->str->exec));
        }
        if(method==0)n.gpu->set_resile(n.initial_step);
        n.optimizer=method;n.adam_step=0;
    }
    Buffer metadata(NetRecord& n){
        Buffer b;b.put<uint32_t>(1);b.array(n.neurons);b.array(n.linear);b.put<int32_t>(n.optimizer);
        b.put<uint64_t>(n.updates);b.put<uint64_t>(n.adam_step);b.put<uint64_t>(n.stage_done);b.put<uint64_t>(n.remaining);
        b.put<float>(n.learning_rate);b.put<float>(n.initial_step);b.string(n.stage);
        return b;
    }
    Buffer snapshot(NetRecord& n){
        checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));
        Buffer b=metadata(n);
        for(layer* l=n.gpu->layers;l->W_plus;l=l->next){
            weights& w=*l->W_plus;b.put<int32_t>(w.act_grad);
            for(int j=0;j<4;++j){std::vector<float> a(w.Wh*(w.Ww+1));checkCudaErrors(cudaMemcpy(a.data(),w.W[j]->mem->vals,a.size()*sizeof(float),cudaMemcpyDeviceToHost));b.array(a);}
        }return b;
    }
    void restore(NetRecord& n,Buffer& b){
        require(b.get<uint32_t>()==1,"unsupported state version");
        require(b.array<int>()==n.neurons&&b.array<int>()==n.linear,"state architecture mismatch");
        int optimizer=b.get<int32_t>();require(optimizer>=-1&&optimizer<=1,"invalid optimizer state");
        uint64_t updates=b.get<uint64_t>(),adam=b.get<uint64_t>(),done=b.get<uint64_t>(),remaining=b.get<uint64_t>();
        float lr=b.get<float>(),step=b.get<float>();auto stage=b.string();
        require(std::isfinite(lr)&&lr>0&&std::isfinite(step)&&step>0,"invalid optimizer settings");
        std::vector<int> active;std::vector<std::vector<float>> arrays;
        for(layer* l=n.gpu->layers;l->W_plus;l=l->next){
            int a=b.get<int32_t>();require(a==1||a==2,"invalid active gradient buffer");active.push_back(a);
            for(int j=0;j<4;++j){auto v=b.array<float>();require(v.size()==size_t(l->W_plus->Wh)*(l->W_plus->Ww+1),"state buffer shape mismatch");arrays.push_back(std::move(v));}
        }b.end();
        checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));int k=0;
        for(layer* l=n.gpu->layers;l->W_plus;l=l->next,++k){l->W_plus->act_grad=active[k];for(int j=0;j<4;++j)checkCudaErrors(cudaMemcpy(l->W_plus->W[j]->mem->vals,arrays[k*4+j].data(),arrays[k*4+j].size()*sizeof(float),cudaMemcpyHostToDevice));}
        n.optimizer=optimizer;n.updates=updates;n.adam_step=adam;n.stage_done=done;n.remaining=remaining;n.learning_rate=lr;n.initial_step=step;n.stage=stage;
    }
    template<bool CollectLoss>
    void pass(NetRecord& n,CostRecord& c,int cost_index,bool backward,bool& first,const double* multiplier=nullptr,
              int begin=0,int end=INT_MAX,bool reuse_forward=false,bool backward_storage=true){
        if(!c.input_id)return;
        backward=backward&&c.has_gradient;backward_storage=backward_storage&&c.has_gradient;
        auto& input=*dat(c.input_id).gpu;network& m=*n.gpu;
        end=std::min(end,input.w);
        require(begin>=0&&begin%c.width==0&&(!reuse_forward||int64_t(end)-begin<=c.width),"invalid cost pass range");
        for(int offset=begin;offset<end;offset+=c.width){
            int valid=std::min(c.width,input.w-offset);
            layer* out;
            if(reuse_forward){
                require(n.loaded_id==c.input_id&&n.loaded_version==dat(c.input_id).version&&
                        n.loaded_width==c.width&&n.loaded_offset==offset&&n.loaded_valid==valid&&n.loaded_rows==c.rows,
                        "forward context is no longer current");
                out=m.layers;while(out->next)out=out->next;
            }else{
                activate(n,c.input_id,c.schema.get(),c.rows,c.width,offset,valid,backward_storage);
                if(c.has_components)out=m.feedforward();
                else {out=m.layers;while(out->next)out=out->next;}
            }
            context& ctx=*out->env;
            int blocks=cpde_reduce::blocks_for(c.width);
            if(c.kind==2){
                c.program->launch(ctx.ADDR->dev,ctx.ADDR_dE?ctx.ADDR_dE->dev:nullptr,c.fields.ptr,c.quadrature_weights.ptr,c.sample_values.ptr,c.sample_errors.ptr,ctx.h,ctx.S->h,input.w,c.width,valid,offset,multiplier,CollectLoss,backward,*m.str->exec);
                CPDE_LAUNCH(cpde_graph::sample_blocks, blocks, cpde_reduce::threads, 0, *m.str->exec, c.sample_values.ptr,c.sample_errors.ptr,c.width,valid,offset,n.reductions.cost_blocks);
            }
            else if(c.kind==0)CPDE_LAUNCH(matching_cost_kernel<CollectLoss>, blocks, cpde_reduce::threads, 0, *m.str->exec, ctx.ADDR->dev,ctx.ADDR_dE?ctx.ADDR_dE->dev:nullptr,dat(c.target_id).gpu->ADDR->dev,c.target_rows.ptr,c.weights.ptr,n.reductions.cost_blocks,ctx.h,ctx.S->h,c.width,valid,offset,c.weight_samples,backward);
            else CPDE_LAUNCH(polynomial_cost_kernel<CollectLoss>, blocks, cpde_reduce::threads, 0, *m.str->exec, ctx.ADDR->dev,ctx.ADDR_dE?ctx.ADDR_dE->dev:nullptr,c.terms.ptr,c.coefficients.ptr,c.terms.size,c.rhs.ptr,c.weights.ptr,n.reductions.cost_blocks,ctx.h,ctx.S->h,c.equations,input.w,c.width,valid,offset,backward);
            checkCudaErrors(cudaGetLastError());
            n.reductions.finish_chunk(blocks,cost_index,*m.str->exec);
            if(backward&&multiplier&&c.kind!=2){CPDE_LAUNCH(cpde_graph::scale_gradients, blocks, cpde_reduce::threads, 0, *m.str->exec, ctx.ADDR_dE->dev,ctx.h,ctx.S->h,c.width,multiplier);checkCudaErrors(cudaGetLastError());}
            if(backward){m.backprop(first?'n':'+');first=false;}
        }
    }
    void prepare_costs(NetRecord& n,const std::vector<uint64_t>& ids,uint64_t ni,bool outputs=false,bool backward=true){
        int need=n.capacity,scratch=1,width=1;
        for(auto id:ids){auto& c=lookup(costs,id);require(c.network_id==ni,"cost belongs to another network");
            if(c.input_id){
                int slots=checked_size(c.width,int(c.rows.size()));
                need=std::max(need,checked_size(slots,backward&&c.has_gradient?2:1));
                scratch=std::max(scratch,slots);width=std::max(width,c.width);
                if(outputs){int full=checked_size(c.width,dat(c.input_id).gpu->S->h);need=std::max(need,full);scratch=std::max(scratch,full);}
            }
            if(c.program)c.program->compile();
        }
        reserve(n,need,scratch,width);
        for(auto id:ids){auto& c=lookup(costs,id);if(c.input_id)prepare(n,c.schema.get(),c.width,backward&&c.has_gradient);}
        n.reductions.reserve(ids.size(),n.neurons.size()-1,*n.gpu->str->exec);
    }
    void graph_pass(NetRecord& n,const std::vector<uint64_t>& ids,GraphRecord& graph,bool backward){
        if(!backward)for(auto id:ids){auto& c=lookup(costs,id);if(c.input_id)prepare(n,c.schema.get(),c.width,false);}
        auto stream=*n.gpu->str->exec;n.reductions.begin(ids.size(),stream);bool first=true;
        int reuse=-1,last_offset=0;double best_work=-1;
        if(backward){
            for(layer* l=n.gpu->layers;l->W_plus;l=l->next){auto& w=*l->W_plus;checkCudaErrors(cudaMemsetAsync(w.act_grad_W()->mem->vals,0,w.Wh*(w.Ww+1)*sizeof(float),stream));}
            // Save the most expensive eligible chunk's forward by running it
            // last. This is a work estimate, not measured kernel timing.
            for(size_t i=0;i<ids.size();++i)if(!graph.schedule[i].known){
                auto& c=lookup(costs,ids[i]);if(!c.input_id||!c.has_gradient||!c.has_components)continue;
                double derivative_work=0,work=0;
                for(int r=0;r<c.schema->h;++r)derivative_work+=std::ldexp(1.0,c.schema->host[r]);
                for(layer* l=n.gpu->layers;l->next;l=l->next){
                    work+=2.0*l->n*l->next->n*c.rows.size();
                    if(l->lin==0)work+=l->n*derivative_work;
                }
                work*=c.width;
                if(work>best_work){best_work=work;reuse=int(i);}
            }
        }
        // Fixed multipliers permit forward, then backward, before switching
        // contexts. Other leaves first contribute only their scalar values.
        for(size_t i=0;i<ids.size();++i){
            if(int(i)==reuse)continue;
            bool immediate=backward&&graph.schedule[i].known&&graph.schedule[i].value!=0;
            pass<true>(n,lookup(costs,ids[i]),i,immediate,first,immediate?graph.fixed_multipliers.ptr+i:nullptr,0,INT_MAX,false,backward);
        }
        if(reuse>=0){
            auto& c=lookup(costs,ids[reuse]);
            last_offset=((dat(c.input_id).gpu->w-1)/c.width)*c.width;
            pass<true>(n,c,reuse,false,first);
        }
        graph.run(n.reductions.cost_totals,ids.size(),backward,stream);
        if(backward){
            // The scalar graph does not touch layer activations. Consume the
            // final forward immediately, before another context overwrites it.
            if(reuse>=0)pass<false>(n,lookup(costs,ids[reuse]),reuse,true,first,
                                   graph.multipliers.ptr+reuse,last_offset,INT_MAX,true);
            // Earlier forward intermediates have been overwritten. Recompute
            // the remaining chunks of leaves with value-dependent multipliers.
            for(size_t i=0;i<ids.size();++i)if(!graph.schedule[i].known)
                pass<false>(n,lookup(costs,ids[i]),i,true,first,graph.multipliers.ptr+i,
                            0,int(i)==reuse?last_offset:INT_MAX);
        }
        graph.queue_download(stream);
    }
    void report_values(NetRecord& n,uint64_t completed,int after,const std::vector<double>& values,GraphRecord* graph){
        Buffer r;r.put<uint64_t>(completed);r.put<uint64_t>(n.updates);r.put<int32_t>(after);r.array(values);
        if(graph)r.put<double>(graph->host.value);wire::send(graph?6:2,r);
    }
    void inspect_gradients(NetRecord& n){
        int index=0;
        for(layer* l=n.gpu->layers;l->W_plus;l=l->next,++index){
            weights& w=*l->W_plus;
            n.reductions.inspect_gradient(w.act_grad_W()->mem->vals,
                checked_size(w.Wh,w.Ww+1),index,*n.gpu->str->exec);
        }
    }
    void gradient_report(NetRecord& n,uint64_t epoch){
        Buffer r;r.put<uint64_t>(epoch);r.put<uint64_t>(n.updates);
        r.put<int32_t>(n.neurons.size()-1);
        for(size_t i=0;i+1<n.neurons.size();++i){
            const auto& g=n.reductions.host_gradients[i];
            r.put<double>(g.squared_norm);r.put<double>(g.max_abs);
            r.put<uint64_t>(g.nonfinite_count);r.put<int32_t>(g.first_nonfinite==INT_MAX?-1:g.first_nonfinite);
        }
        wire::send(5,r);
    }
    void validate_costs(NetRecord& n,const std::vector<uint64_t>& ids,std::vector<double>& values,bool before_update=true){
        const std::string phase=before_update?"; no optimizer update performed for this iteration":" during objective evaluation";
        for(size_t i=0;i<ids.size();++i){
            const auto& c=n.reductions.host_costs[i];values[i]=c.value;
            if(c.bad_sample!=INT_MAX){
                std::string what=c.bad_component<0?"objective":"dE/dY component "+std::to_string(c.bad_component);
                throw std::runtime_error("non-finite "+what+" at cost index "+std::to_string(i)+
                    ", sample "+std::to_string(c.bad_sample)+phase);
            }
            if(!std::isfinite(c.value))throw std::runtime_error("non-finite reduced objective"+phase);
        }
    }
    void evaluate_costs(NetRecord& n,const std::vector<uint64_t>& ids,std::vector<double>& values,GraphRecord* graph=nullptr){
        if(graph){graph_pass(n,ids,*graph,false);n.reductions.download(ids.size(),0,*n.gpu->str->exec);validate_costs(n,ids,values,false);graph->validate();return;}
        n.reductions.begin(ids.size(),*n.gpu->str->exec);
        bool unused=true;
        for(size_t i=0;i<ids.size();++i){auto& c=lookup(costs,ids[i]);
            if(c.input_id)prepare(n,c.schema.get(),c.width,false);
            pass<true>(n,c,i,false,unused,nullptr,0,INT_MAX,false,false);
        }
        n.reductions.download(ids.size(),0,*n.gpu->str->exec);
        validate_costs(n,ids,values,false);
    }
    void train(Buffer& b,std::atomic<bool>& stop,bool graph_mode=false){
        uint64_t ni=b.get<uint64_t>();auto& n=net(ni);auto ids=b.array<uint64_t>();uint64_t epochs=b.get<uint64_t>();
        int method=b.get<int32_t>(),fresh=b.get<int32_t>();float steps=b.get<float>();int moments=b.get<int32_t>();float lr=b.get<float>();
        int verbose=b.get<int32_t>(),log=b.get<int32_t>(),eval=b.get<int32_t>(),outputs=b.get<int32_t>(),save=b.get<int32_t>();auto stage=b.string();int diagnostics=b.get<int32_t>();
        std::unique_ptr<GraphRecord> graph;if(graph_mode){graph.reset(new GraphRecord);graph->read(b,ids.size());}
        // Optional capability-gated trailer; old clients keep their existing wire layout.
        uint64_t restore_every=0;float restore_step=0;
        if(b.pos<b.bytes.size()){restore_every=b.get<uint64_t>();restore_step=b.get<float>();}b.end();
        require(!ids.empty()&&epochs>0,"training needs costs and positive epochs");require(method==0||method==1,"unknown optimizer");
        require(ids.size()<=INT_MAX&&log>=0&&eval>=0&&save>=0&&diagnostics>=0&&std::isfinite(lr)&&lr>=0&&std::isfinite(steps)&&steps>=0,"invalid training controls");
        require((restore_every==0&&restore_step==0)||(restore_every>0&&method==0&&std::isfinite(restore_step)&&restore_step>=std::numeric_limits<float>::min()),"invalid periodic RPROP restoration controls");
        prepare_costs(n,ids,ni,outputs);
        if(fresh||n.optimizer!=method)reset(n,method);
        if(lr>0)n.learning_rate=lr;
        if(steps>0){require(method==0,"reset_steps requires RPROP");n.gpu->set_resile(steps);}
        if(moments){require(method==1,"reset_moments requires Adam");reset(n,1);}
        if(graph){if(n.stage.substr(0,n.stage.find('|'))!=stage.substr(0,stage.find('|')))n.stage_done=0;n.stage=stage;}
        else if(n.stage!=stage){n.stage=stage;n.stage_done=0;}
        n.remaining=epochs;
        std::vector<double> values(ids.size());uint64_t completed=0,last_saved=UINT64_MAX,last_eval=UINT64_MAX;
        for(;completed<epochs&&!stop.load();){
            const bool collect=(log&&(completed+1)%log==0)||completed+1==epochs;
            if(graph)graph_pass(n,ids,*graph,true);
            else {n.reductions.begin(ids.size(),*n.gpu->str->exec);
            bool first=true;
            for(size_t i=0;i<ids.size();++i){
                if(collect)pass<true>(n,lookup(costs,ids[i]),i,true,first);
                else pass<false>(n,lookup(costs,ids[i]),i,true,first);
            }
            }
            inspect_gradients(n);
            // One validation boundary for all costs/chunks, before touching optimizer state.
            n.reductions.download(ids.size(),n.neurons.size()-1,*n.gpu->str->exec);
            bool bad_gradient=false;
            for(size_t i=0;i+1<n.neurons.size();++i)
                bad_gradient=bad_gradient||n.reductions.host_gradients[i].nonfinite_count!=0;
            if(bad_gradient||(diagnostics&&((completed+1)%diagnostics==0||completed+1==epochs)))
                gradient_report(n,completed+1);
            validate_costs(n,ids,values);
            if(graph)graph->validate();
            if(bad_gradient){
                for(size_t i=0;i+1<n.neurons.size();++i){
                    const auto& g=n.reductions.host_gradients[i];
                    if(g.nonfinite_count)throw std::runtime_error("non-finite weight gradient in layer "+std::to_string(i)+
                        ", flattened index "+std::to_string(g.first_nonfinite)+"; no optimizer update performed for this iteration");
                }
            }
            if(method==0)n.gpu->train();else {require(n.adam_step<uint64_t(INT_MAX),"Adam counter exceeds supported range");n.gpu->train(int(++n.adam_step),n.learning_rate);}
            checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));
            ++completed;++n.updates;++n.stage_done;n.remaining=epochs-completed;
            // The absolute update count survives short calls, energy updates and checkpoints.
            if(restore_every&&n.updates%restore_every==0){
                n.gpu->reset_resile(restore_step);
                checkCudaErrors(cudaGetLastError());checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));
            }
            bool last=completed==epochs||stop.load();
            if(collect)report_values(n,completed,0,values,graph.get());
            if(eval&&(completed%eval==0||last)){
                evaluate_costs(n,ids,values,graph.get());
                report_values(n,completed,1,values,graph.get());last_eval=completed;
                if(outputs)for(auto id:ids){auto& c=lookup(costs,id);uint64_t output=evaluate(ni,c.input_id,c.width);auto& d=*dat(output).gpu;std::vector<float> a(d.mem->size);checkCudaErrors(cudaMemcpy(a.data(),d.mem->vals,a.size()*sizeof(float),cudaMemcpyDeviceToHost));Buffer r;r.put<uint64_t>(id);r.array(a);data.erase(output);wire::send(4,r);}
            }
            if(save&&(completed%save==0||last)){wire::send(3,snapshot(n));last_saved=completed;}
        }
        if(last_eval!=completed){
        evaluate_costs(n,ids,values,graph.get());
        report_values(n,completed,1,values,graph.get());
        }
        // State stays on GPU unless checkpointing or an explicit export was requested.
        if(save&&last_saved!=completed)wire::send(3,snapshot(n));
        Buffer result;result.put<uint64_t>(completed);result.put<int32_t>(completed<epochs);auto meta=metadata(n);result.array(meta.bytes);wire::send(0,result);
    }
    void command(uint32_t op,Buffer& b,std::atomic<bool>& stop){
        Buffer out;
        if(op==1){b.end();out.string("CPDE/3");out.put<int32_t>(6);
#if defined(CPDE_CPU_BACKEND)
            out.put<uint32_t>(63); // existing capabilities plus production CPU backend
#elif defined(CPDE_HOST_PROBE)
            out.put<uint32_t>(95); // existing capabilities plus test-only host probe
#elif defined(CPDE_ENABLE_NVRTC)
            out.put<uint32_t>(31); // diagnostics, graphs, expressions, RPROP restore, wrapper initialization
#else
            out.put<uint32_t>(27);
#endif
        }
        else if(op==2){
            int d=b.get<int32_t>(),h=b.get<int32_t>(),w=b.get<int32_t>(),v=b.get<int32_t>();checked_size(d,h,w);require(v>0&&v<=26,"invalid variable count");
            auto rows=b.array<int>();auto a=b.array<float>();b.end();require(rows.size()==size_t(d)*(v+1)&&a.size()==size_t(d)*h*w,"data shape mismatch");
            itable s(d,v+1);for(int r=0;r<d;++r)for(int c=0;c<=v;++c)s.host[c*d+r]=rows[r*(v+1)+c];s.todev();
            auto rec=std::unique_ptr<DataRecord>(new DataRecord);rec->gpu.reset(new inout(h,w,&s,device));checkCudaErrors(cudaMemcpy(rec->gpu->mem->vals,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
            uint64_t id=next_id++;data[id]=std::move(rec);out.put<uint64_t>(id);
        }else if(op==3){
            auto ns=b.array<int>(),ls=b.array<int>();require(ns.size()>=2&&ns.size()<=256&&ns.size()==ls.size(),"invalid architecture");
            for(size_t i=0;i<ns.size();++i)require(ns[i]>0&&(ls[i]==0||ls[i]==1),"invalid layer");require(ls.front()==1&&ls.back()==1,"input and output layers must be linear");
            require(b.pos<b.bytes.size(),"initial weights required; update the Python wrapper");
            std::vector<std::vector<float>> initial;
            for(size_t i=1;i<ns.size();++i){
                auto values=b.array<float>();
                require(values.size()==size_t(ns[i])*(size_t(ns[i-1])+1),"initial weight shape mismatch");
                for(float value:values)require(std::isfinite(value),"nonfinite initial weight");
                initial.push_back(std::move(values));
            }
            b.end();
            conf c;c.layers=ns.size();c.neurons=(int*)malloc(ns.size()*sizeof(int));c.lin=(int*)malloc(ls.size()*sizeof(int));std::copy(ns.begin(),ns.end(),c.neurons);std::copy(ls.begin(),ls.end(),c.lin);c.maxlen=2;c.devID=device;
            auto r=std::unique_ptr<NetRecord>(new NetRecord);r->neurons=ns;r->linear=ls;r->gpu.reset(new network(&c));
            size_t layer_index=0;
            for(layer* l=r->gpu->layers;l->W_plus;l=l->next,++layer_index)
                checkCudaErrors(cudaMemcpy(l->W_plus->W[0]->mem->vals,initial[layer_index].data(),
                    initial[layer_index].size()*sizeof(float),cudaMemcpyHostToDevice));
            uint64_t id=next_id++;nets[id]=std::move(r);out.put<uint64_t>(id);
        }else if(op==4){auto ni=b.get<uint64_t>(),di=b.get<uint64_t>();int chunk=b.get<int32_t>();b.end();out.put<uint64_t>(evaluate(ni,di,chunk));}
        else if(op==5){auto& d=*dat(b.get<uint64_t>()).gpu;b.end();std::vector<float> a(d.mem->size);checkCudaErrors(cudaMemcpy(a.data(),d.mem->vals,a.size()*sizeof(float),cudaMemcpyDeviceToHost));out.array(a);}
        else if(op==6){auto& d=dat(b.get<uint64_t>());auto a=b.array<float>();b.end();require(a.size()==size_t(d.gpu->mem->size),"replacement data shape mismatch");checkCudaErrors(cudaMemcpy(d.gpu->mem->vals,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));++d.version;}
        else if(op==7){
            auto c=std::unique_ptr<CostRecord>(new CostRecord);c->network_id=b.get<uint64_t>();c->input_id=b.get<uint64_t>();c->target_id=b.get<uint64_t>();c->kind=b.get<int32_t>();int chunk=b.get<int32_t>();c->rows=b.array<int>();
            auto& n=net(c->network_id);auto& d=*dat(c->input_id).gpu;require(d.h==n.neurons.front(),"input channel mismatch");c->width=chunk?std::min(chunk,d.w):d.w;require(c->width>0,"invalid chunk size");c->schema=make_schema(d,c->rows);
            if(c->kind==0){auto& t=*dat(c->target_id).gpu;require(t.h==n.neurons.back()&&t.w==d.w,"target shape mismatch");auto tr=b.array<int>();c->weight_samples=b.get<int32_t>();auto w=b.array<float>();require(tr.size()==c->rows.size()&&(c->weight_samples==1||c->weight_samples==d.w)&&w.size()==c->rows.size()*c->weight_samples*t.h,"matching weight shape mismatch");for(int r:tr)require(r>=0&&r<t.S->h,"target derivative mismatch");for(float x:w)require(std::isfinite(x)&&x>=0,"invalid matching weight");c->target_rows.upload(tr);c->weights.upload(w);}
            else{require(c->kind==1,"unknown cost kind");c->equations=b.get<int32_t>();auto raw=b.array<int>();auto co=b.array<float>(),rhs=b.array<float>(),w=b.array<float>();require(c->equations>0&&raw.size()%3==0&&co.size()==raw.size()/3*d.w&&rhs.size()==size_t(c->equations)*d.w&&w.size()==rhs.size(),"equation data shape mismatch");std::vector<PolynomialTerm> terms;for(size_t i=0;i<raw.size();i+=3){require(raw[i]>=0&&raw[i]<c->equations&&raw[i+1]>=0&&raw[i+1]<int(c->rows.size())*n.neurons.back()&&raw[i+2]>=-1&&raw[i+2]<int(c->rows.size())*n.neurons.back(),"invalid polynomial term");terms.push_back({raw[i],raw[i+1],raw[i+2]});}for(float x:w)require(std::isfinite(x)&&x>=0,"invalid equation weight");for(float x:co)require(std::isfinite(x),"invalid coefficient");for(float x:rhs)require(std::isfinite(x),"invalid right-hand side");c->terms.upload(terms);c->coefficients.upload(co);c->rhs.upload(rhs);c->weights.upload(w);}
            b.end();uint64_t id=next_id++;costs[id]=std::move(c);out.put<uint64_t>(id);
        }else if(op==8){train(b,stop);return;}
        else if(op==9){auto& n=net(b.get<uint64_t>());b.end();out=snapshot(n);}
        else if(op==10){auto& n=net(b.get<uint64_t>());restore(n,b);}
        else if(op==11){int kind=b.get<int32_t>();auto id=b.get<uint64_t>();b.end();
            if(kind==0){for(auto& c:costs)require(c.second->input_id!=id&&c.second->target_id!=id,"data is retained by a cost; close that cost first");data.erase(id);}
            else if(kind==1){for(auto& c:costs)require(c.second->network_id!=id,"network is retained by a cost; close that cost first");nets.erase(id);}else if(kind==2)costs.erase(id);else require(false,"invalid object kind");
        }else if(op==12){
            auto c=std::unique_ptr<CostRecord>(new CostRecord);c->kind=2;c->target_id=0;
            c->network_id=b.get<uint64_t>();c->input_id=b.get<uint64_t>();int chunk=b.get<int>();c->samples=b.get<int>();
            auto raw=b.array<int>();auto constants=b.array<double>();int fields=b.get<int>();auto numeric=b.array<double>(),weights=b.array<double>();b.end();
            auto& n=net(c->network_id);
            require(c->samples>=0&&chunk>=0&&fields>=0&&fields<=512&&weights.size()==size_t(c->samples)&&numeric.size()==size_t(fields)*c->samples,"invalid integral field shape");
            require(!constants.empty()&&constants.size()<=512&&raw.size()==constants.size()*3,"invalid expression IR size");
            if(c->samples){auto& d=*dat(c->input_id).gpu;require(d.h==n.neurons.front()&&d.w==c->samples,"integral input shape mismatch");
                std::vector<int> requested;
                for(size_t i=0;i<constants.size();++i)if(raw[3*i]==1){
                    int component=raw[3*i+1];
                    require(component>=0&&int64_t(component)<int64_t(d.S->h)*n.neurons.back(),"component outside derivative schema");
                    requested.push_back(component/n.neurons.back());
                }
                for(int row:all_rows(d)){
                    bool needed=d.S->host[row]==0;
                    for(int target:requested){
                        bool dependency=true;
                        for(int v=1;v<d.S->w;++v)if(d.S->host[v*d.S->h+row]>d.S->host[v*d.S->h+target]){dependency=false;break;}
                        needed=needed||dependency;
                    }
                    if(needed)c->rows.push_back(row);
                }
                c->schema=make_schema(d,c->rows);c->width=chunk?std::min(chunk,d.w):d.w;
            }else {require(!c->input_id,"empty integral must not have input data");c->width=0;}
            std::vector<cpde_scalar::Node> ir;
            for(int i=0;i<int(constants.size());++i){int op=raw[3*i],a=raw[3*i+1],mask=raw[3*i+2];
                require(op>=0&&op<=14&&std::isfinite(constants[i]),"invalid expression node");
                if(op==1){require(a>=0&&(mask==0||mask==1),"invalid component");if(c->samples){
                    auto row=std::find(c->rows.begin(),c->rows.end(),a/n.neurons.back());
                    require(row!=c->rows.end(),"component outside derivative closure");
                    a=int(row-c->rows.begin())*n.neurons.back()+a%n.neurons.back();
                }}
                if(op==2)require(a>=0&&a<fields,"invalid field index");
                if(op>=3){require(a>=0&&a<i,"invalid expression dependency");if(op<=6)require(mask>=0&&mask<i,"invalid expression dependency");}
                ir.push_back({op,a,mask,constants[i]});
            }
            for(double x:numeric)require(std::isfinite(x),"nonfinite integral field");for(double x:weights)require(std::isfinite(x),"nonfinite quadrature weight");
            c->has_components=false;c->has_gradient=false;
            for(const auto& node:ir)if(node.op==1){c->has_components=true;if(node.b==0)c->has_gradient=true;}
            auto source=cpde_expression::source(ir);auto found=programs.find(source);
            if(found==programs.end())found=programs.emplace(source,std::make_shared<cpde_expression::Program>(std::move(ir))).first;
            c->program=found->second;c->fields.upload(numeric);c->quadrature_weights.upload(weights);
            c->sample_values.upload(std::vector<double>(c->width));c->sample_errors.upload(std::vector<int>(c->width));
            uint64_t id=next_id++;costs[id]=std::move(c);out.put<uint64_t>(id);
        }else if(op==13){auto& c=lookup(costs,b.get<uint64_t>());bool compile=b.get<int>()!=0;b.end();require(bool(c.program),"not a compiled expression cost");if(compile)c.program->compile();out.string(c.program->code);out.string(c.program->log);}
        else if(op==14){
            auto ni=b.get<uint64_t>();auto& n=net(ni);auto ids=b.array<uint64_t>();bool gradient=b.get<int>()!=0;
            require(!ids.empty(),"cost graph needs at least one integral");GraphRecord graph;graph.read(b,ids.size());b.end();prepare_costs(n,ids,ni,false,gradient);
            Buffer saved;if(gradient)saved=snapshot(n);
            try{
                std::vector<double> values(ids.size());graph_pass(n,ids,graph,gradient);
                if(gradient)inspect_gradients(n);
                n.reductions.download(ids.size(),gradient?n.neurons.size()-1:0,*n.gpu->str->exec);
                validate_costs(n,ids,values,false);graph.validate();out.put<double>(graph.host.value);
                if(gradient){out.put<int>(n.neurons.size()-1);int index=0;
                    for(layer* l=n.gpu->layers;l->W_plus;l=l->next,++index){
                        require(!n.reductions.host_gradients[index].nonfinite_count,"nonfinite weight gradient during evaluation");
                        auto& w=*l->W_plus;std::vector<float> g(w.Wh*(w.Ww+1));checkCudaErrors(cudaMemcpy(g.data(),w.act_grad_W()->mem->vals,g.size()*sizeof(float),cudaMemcpyDeviceToHost));out.array(g);
                    }
                }
            }catch(...){if(gradient)restore(n,saved);throw;}
            if(gradient)restore(n,saved);
        }else if(op==15){train(b,stop,true);return;}
        else require(false,"unknown command");
        wire::send(0,out);
    }
};
int main(int argc,char** argv){
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    std::cout.rdbuf(std::cerr.rdbuf());int device=0;
    if(argc==3&&std::string(argv[1])=="--device")device=std::stoi(argv[2]);
    std::mutex mutex;std::condition_variable ready;std::deque<Buffer> queue;
    bool eof=false;std::atomic<bool> stop(false),training(false);std::string input_error;
    std::thread reader([&]{try{for(;;){Buffer b;if(!wire::read(b))break;uint32_t op=b.get<uint32_t>();
        if(op==99){if(training.load())stop.store(true);continue;}
        if(op==8||op==15){stop.store(false);training.store(true);}
        bool quit=op==0;b.pos=0;{std::lock_guard<std::mutex> lock(mutex);queue.push_back(std::move(b));}ready.notify_one();if(quit)break;
    }}catch(const std::exception& e){input_error=e.what();}
    {std::lock_guard<std::mutex> lock(mutex);eof=true;}stop.store(true);ready.notify_one();});
    try{Server server(device);for(;;){Buffer b;{std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return eof||!queue.empty();});if(queue.empty())break;b=std::move(queue.front());queue.pop_front();}
        uint32_t op=b.get<uint32_t>();if(op==0){wire::send(0);break;}
        try{server.command(op,b,stop);}catch(const std::exception& e){Buffer err;err.string(e.what());wire::send(1,err);}if(op==8||op==15)training.store(false);stop.store(false);
    }}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;}
    reader.join();return input_error.empty()?0:1;
}
