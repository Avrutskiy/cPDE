// Internal resource invariants, also built by the serial host test helper.
#include <cstring>
#define main cpde_server_entry_for_probe
#include "cpde_server.cu"
#undef main
static void expect(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){
    try {
        checkCudaErrors(cudaSetDevice(0));
        for(char location: {'h','d'}) {
            matrix source(2,2,location);source.init(3);
            matrix copy(&source,location);matrix host(&copy,'h');
            for(int i=0;i<4;++i)expect(host.mem->vals[i]==3,"matrix copy data mismatch");
        }
        {
            memory* undersized=new memory(4,'h');bool rejected=false;
            try {matrix invalid(undersized,2,4);}catch(const std::invalid_argument&){rejected=true;}
            expect(rejected,"oversized matrix view accepted");delete undersized;
            matrix valid(new memory(8,'h'),2,2);expect(valid.mem!=nullptr,"valid matrix view rejected");
        }
        Server server(0);NetRecord n;
        n.neurons={1,4,3,1};n.linear={1,0,0,1};
        conf config;config.layers=4;config.devID=0;config.maxlen=2;
        config.neurons=(int*)malloc(4*sizeof(int));config.lin=(int*)malloc(4*sizeof(int));
        std::copy(n.neurons.begin(),n.neurons.end(),config.neurons);std::copy(n.linear.begin(),n.linear.end(),config.lin);
        n.gpu.reset(new network(&config));
        itable schema(3,2);int entries[]={0,1,2,0,1,2};std::copy(entries,entries+6,schema.host);schema.todev();
        server.prepare(n,&schema,7,false);
        expect(n.capacity==21,"forward-only activation storage is doubled");
        expect(n.gpu->ones->w==7,"ones includes derivative rows");
        expect(n.gpu->cache->mem->size==84,"forward jet scratch size mismatch");
        for(layer* l=n.gpu->layers;l;l=l->next){
            context* c=l->get_context(&schema,l->n,7,0);
            expect(c&&!c->ADDR_dE&&!c->tables,"forward context allocated backward tables");
            if(l->lin)expect(!c->STCK,"linear context allocated nonlinear stack");
        }
        server.prepare(n,&schema,7,true);
        expect(n.capacity==42,"backward storage missing");
        auto first=n.gpu->layers->next->get_context(&schema,4,7,1)->tables;
        auto second=n.gpu->layers->next->next->get_context(&schema,3,7,1)->tables;
        expect(first&&first==second,"metadata not shared across layers");
        server.prepare(n,&schema,3,true);
        auto narrow=n.gpu->layers->next->get_context(&schema,4,3,1)->tables;
        expect(first==narrow,"metadata not shared across widths");
        expect(n.gpu->ones->w==7,"ones unnecessarily grew");
        checkCudaErrors(cudaStreamSynchronize(*n.gpu->str->exec));
        printf("PASS: matrix copy/views; forward storage=21, backward storage=42, scratch=84, ones=7; shared metadata across layers/widths\n");
    } catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
