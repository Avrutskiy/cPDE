#pragma once
#include "backend_launch.hpp"
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "derivative_kernels_v4.cu"


#include "cuda_check_v1.cuh"


class stream {
public:
    cudaStream_t* exec;
    stream() {
        exec=(cudaStream_t*)malloc(sizeof(cudaStream_t));
        checkCudaErrors(cudaStreamCreate(exec));
    }
    ~stream() {checkCudaErrors(cudaStreamDestroy(*exec));free(exec);}
};

class memory { // storage for matrices
public:
	char loc; // 'd' = device, 'h' = host, 'N' = non-allocated
	int dev; // device ID
	int size; // total amount of floats
	float* vals;
	int alloc;

	int transfer(memory* target) {

		long int length;
		if (this->size > target->size) length = target->size;
		else length = this->size;

		if (this->loc == 'N' || target->loc == 'N')	return -1;

		if (target->loc == 'h') {
			if (this->loc == 'h') checkCudaErrors(cudaMemcpy(target->vals, this->vals, length * sizeof(float), cudaMemcpyHostToHost));
			if (this->loc == 'd') checkCudaErrors(cudaMemcpy(target->vals, this->vals, length * sizeof(float), cudaMemcpyDeviceToHost));
		}
		if (target->loc == 'd') {
			if (this->loc == 'h') checkCudaErrors(cudaMemcpy(target->vals, this->vals, length * sizeof(float), cudaMemcpyHostToDevice));
			if (this->loc == 'd') checkCudaErrors(cudaMemcpy(target->vals, this->vals, length * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		return length;
	}
	memory() {
		loc = 'N';
		dev = -1;
		vals = NULL;
		size = 0;
	}
	~memory() {
		if (loc == 'h' && alloc == 1) checkCudaErrors(cudaFreeHost(vals));
		if (loc == 'd' && alloc == 1) checkCudaErrors(cudaFree(vals));
		vals = NULL;
		dev = -1;
		loc = 'N';
	}
	memory(int SIZE, char l) {
		if (l == 'h' || l == 'd') loc = l;
		else { loc = 'N'; vals = NULL; dev = -1; alloc = 0; }
		if (loc == 'h') { checkCudaErrors(cudaHostAlloc((void**)&vals, SIZE * sizeof(float), cudaHostAllocDefault)); dev = -1; alloc = 1; }
		if (loc == 'd') { checkCudaErrors(cudaMalloc((void**)&vals, SIZE * sizeof(float))); checkCudaErrors(cudaGetDevice(&dev)); alloc = 1; }
		size = SIZE;
	};
	memory(float* addr, int SIZE, char l) {
		if (l == 'h' || l == 'd') loc = l;
		else { loc = 'N'; vals = NULL; dev = -1; alloc = 0; }
		checkCudaErrors(cudaGetDevice(&dev));
		vals = addr;
		size = SIZE;
		alloc = 0;
	}
};

class matrix {
public:
	int w;
	int h;
	memory* mem;

	matrix() {
		w = 0;
		h = 0;
		mem = NULL;
	}
	matrix(int H, int W, char l) {
		h = H;
		w = W;
		mem = new memory(H*W, l);
	}
	matrix(matrix* src, char l) {
        h = src->h; w = src->w;
        mem = new memory(h*w, l);
        src->mem->transfer(mem);
    }

	matrix(memory* MEM, int H, int W) {
		mem = MEM;
		h = H;
		w = W;
		if (H <= 0 || W <= 0 || int64_t(H)*W > MEM->size)
            throw std::invalid_argument("matrix view exceeds storage capacity");
	}

    void init(float val, char mode = 'v', cudaStream_t on = 0) {
        if(mode!='v')throw std::invalid_argument("random initialization belongs to the Python wrapper");
        if(mem->loc=='d') {
            if(val==0)checkCudaErrors(cudaMemsetAsync(mem->vals,0,size_t(h)*w*sizeof(float),on));
            else {CPDE_LAUNCH(Fill_Constant_k, 64, 64, 0, on, h*w,val,mem->vals);checkCudaErrors(cudaGetLastError());}
        } else if(mem->loc=='h')std::fill(mem->vals,mem->vals+h*w,val);
    }

	~matrix() {
		delete mem;
	}
};

class itable {
public:
	int h;
	int w;
	int* dev;
	int* host;


	void todev() {
		if (host != NULL && dev != NULL) checkCudaErrors(cudaMemcpy(dev, host, h*w * sizeof(int), cudaMemcpyHostToDevice));
	}
	void fromdev() {
		if (host != NULL && dev != NULL) checkCudaErrors(cudaMemcpy(host, dev, h*w * sizeof(int), cudaMemcpyDeviceToHost));
	}
	int el(int x, int y) {
		return host[h*y + x];
	}
	void set_el(int x, int y, int val) {
		host[h*y + x] = val;
		//todev();
	}

	itable(int H, int W) {
		h = H;
		w = W;
		if (h*w > 0) {
			checkCudaErrors(cudaHostAlloc((void**)&host, h*w * sizeof(int), cudaHostAllocDefault));
			checkCudaErrors(cudaMalloc((void**)&dev, h*w * sizeof(int)));
			for (int i = 0; i < h*w; i++) { host[i] = 0; }
		}
		else {
			dev = NULL;
			host = NULL;
		}
		//todev();
	}
	itable(int H, int W, char loc) {
		h = H;
		w = W;
		if (h*w > 0) {
			checkCudaErrors(cudaHostAlloc((void**)&host, h*w * sizeof(int), cudaHostAllocDefault));
			dev = NULL;
			if (loc != 'h') checkCudaErrors(cudaMalloc((void**)&dev, h*w * sizeof(int)));
			for (int i = 0; i < h*w; i++) { host[i] = 0; }
		}
		else {
			dev = NULL;
			host = NULL;
		}
		//todev();
	}
	itable(itable* s) {
		h = s->h;
		w = s->w;
		if (h*w > 0) {
			checkCudaErrors(cudaHostAlloc((void**)&host, h*w * sizeof(int), cudaHostAllocDefault));
			checkCudaErrors(cudaMalloc((void**)&dev, h*w * sizeof(int)));
			if (dev != NULL && s->dev != NULL) checkCudaErrors(cudaMemcpy(dev, s->dev, h*w * sizeof(int), cudaMemcpyDeviceToDevice));
			if (host != NULL && s->host != NULL) checkCudaErrors(cudaMemcpy(host, s->host, h*w * sizeof(int), cudaMemcpyHostToHost));
		}
		else {
			host = NULL;
			dev = NULL;
		}
	}


	itable() {
		h = 0;
		w = 0;
		dev = NULL;
		host = NULL;
	}
	~itable() {
		checkCudaErrors(cudaFreeHost(host));
		checkCudaErrors(cudaFree(dev));
		host = NULL;
		dev = NULL;
	}
};

int SSets(int len, itable* vector, itable* result, int offset) {

	int L = vector->w;
	int H = result->h;
	int iter = offset;

	if (len == 0) {
		iter++;
	}

	if (len == 1) {
		for (int i = 0; i < L; i++) {
			result->host[iter + H * 0] = vector->host[i];
			iter++;
		}
	}

	if (len == 2) {
		for (int i = 0; i < L; i++) {
			for (int j = i + 1; j < L; j++) {
				result->host[iter + H * 0] = vector->host[i];
				result->host[iter + H * 1] = vector->host[j];
				iter++;
			}
		}
	}

	if (len == 3) {
		for (int i = 0; i < L; i++) {
			for (int j = i + 1; j < L; j++) {
				for (int k = j + 1; k < L; k++) {
					result->host[iter + H * 0] = vector->host[i];
					result->host[iter + H * 1] = vector->host[j];
					result->host[iter + H * 2] = vector->host[k];
					iter++;
				}
			}
		}
	}

	if (len == 4) {
		for (int i = 0; i < L; i++) {
			for (int j = i + 1; j < L; j++) {
				for (int k = j + 1; k < L; k++) {
					for (int l = k + 1; l < L; l++) {
						result->host[iter + H * 0] = vector->host[i];
						result->host[iter + H * 1] = vector->host[j];
						result->host[iter + H * 2] = vector->host[k];
						result->host[iter + H * 3] = vector->host[l];
						iter++;
					}
				}
			}
		}
	}

	if (len == 5) {
		for (int i = 0; i < L; i++) {
			for (int j = i + 1; j < L; j++) {
				for (int k = j + 1; k < L; k++) {
					for (int l = k + 1; l < L; l++) {
						for (int m = l + 1; m < L; m++) {
							result->host[iter + H * 0] = vector->host[i];
							result->host[iter + H * 1] = vector->host[j];
							result->host[iter + H * 2] = vector->host[k];
							result->host[iter + H * 3] = vector->host[l];
							result->host[iter + H * 4] = vector->host[m];
							iter++;
						}
					}
				}
			}
		}
	}

	if (len == 6) {
		for (int i = 0; i < L; i++) {
			for (int j = i + 1; j < L; j++) {
				for (int k = j + 1; k < L; k++) {
					for (int l = k + 1; l < L; l++) {
						for (int m = l + 1; m < L; m++) {
							for (int n = m + 1; n < L; n++) {
								result->host[iter + H * 0] = vector->host[i];
								result->host[iter + H * 1] = vector->host[j];
								result->host[iter + H * 2] = vector->host[k];
								result->host[iter + H * 3] = vector->host[l];
								result->host[iter + H * 4] = vector->host[m];
								result->host[iter + H * 5] = vector->host[n];
								iter++;
							}
						}
					}
				}
			}
		}
	}

	return iter;
}

int resolve(itable* S, itable* key, int str) {

	int He = S->h;
	int We = key->w;
	int cm;
	int ck;
	int d1, d2;
	int *Sa, *Sb;

	for (int i = 0; i < He; i++) {
		Sa = S->host + i;
		Sb = key->host + str;

		cm = 1;
		for (int q = 0; q < We; q++) {
			d1 = *(Sa + He * (q + 1));
			d2 = *(Sb + (key->h)*q);
			if (d1 == d2) ck = 1; else ck = 0;
			cm = cm * ck;
		}

		if (cm) {
			return i;
		}

	}
	return -1;
}

class ptable {
public:
	int l;
	int w;
	int h;
	float** dev;
	float** host;

	ptable(int L, int H, int W) {
		l = L;
		h = H;
		w = W;
		checkCudaErrors(cudaHostAlloc((void**)&host, l * sizeof(float*), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc((void**)&dev, l * sizeof(float*)));
	}

	ptable(int L, int H, int W, memory* mem, int mode = 0) {
		if (L > 0 && mem->loc == 'd' && mem->size >= L*H*W) {
			l = L;
			h = H;
			w = W;
			checkCudaErrors(cudaHostAlloc((void**)&host, l * sizeof(float*), cudaHostAllocDefault));
			checkCudaErrors(cudaMalloc((void**)&dev, l * sizeof(float*)));
			for (int i = 0; i < l; i++) host[i] = mem->vals + i * h*w + mode * h*w*l;
			checkCudaErrors(cudaMemcpy(dev, host, l * sizeof(float*), cudaMemcpyHostToDevice));
		}
		else {
			fprintf(stderr, "\tptable was not created\n");
			l = 0;
			h = 0;
			w = 0;
			host = NULL;
			dev = NULL;
		}
	}

	ptable() {
		l = 0;
		w = 0;
		h = 0;
		dev = NULL;
		host = NULL;
	}
	~ptable() {
		checkCudaErrors(cudaFreeHost(host));
		checkCudaErrors(cudaFree(dev));
	}
};

class stack {
public:
	int H;
	int W;
	float** dev;
	float** host;

	stack(ptable* AD, itable* S, itable* dTab, itable* Crs) {
		int l = AD->l;
		int h = AD->h;
		int w = AD->w;

		// 64 is a maximum length of stack for |D|=6; Note that unlike any other, this matrix is written in row-major way.
		checkCudaErrors(cudaHostAlloc((void**)&host, l * 64 * sizeof(float*), cudaHostAllocDefault));
		checkCudaErrors(cudaMalloc((void**)&dev, l * 64 * sizeof(float*)));
		for (int i = 0; i < 64 * l; i++) host[i] = NULL;

		int order;
		itable* subsets = NULL;
		itable* der_vec = NULL;
		itable* compacted = NULL;
		int subs_l = 0;
		int we = S->w;
		int he = S->h;
		int ex;
		int tau;
		int crr = 0;
		int RS;

		for (int i = 0; i < l; i++) {
			order = S->host[i];
			subs_l = pow(2, order);
			subsets = new itable(subs_l, order, 'h');
			der_vec = new itable(1, order, 'h');
			tau = 0;
			for (int q = 1; q < we; q++) {
				ex = S->host[i + q * he];
				while (ex > 0) { der_vec->set_el(0, tau, q); ex--; tau++; }
			}

			crr = 0;

			for (int om = 0; om <= order; om++) {
				crr = SSets(om, der_vec, subsets, crr);
			}

			compacted = new itable(subsets->h, S->w - 1, 'h');

			for (int i = 0; i < compacted->h; i++) {
				for (int j = 0; j < subsets->w; j++) {
					if (subsets->el(i, j) != 0) compacted->host[(subsets->el(i, j) - 1)*compacted->h + i]++;
				}
			}

			for (int ay = 0; ay < compacted->h; ay++) {
				RS = resolve(S, compacted, ay);
				host[64 * i + ay] = *(AD->host + RS);
			}

			delete(der_vec);
			delete(subsets);
			delete(compacted);
		}
		checkCudaErrors(cudaMemcpy(dev, host, l * 64 * sizeof(float*), cudaMemcpyHostToDevice));
		H = S->h;
		W = 64;
	}

	stack() {
		W = 0;
		H = 0;
		dev = NULL;
		host = NULL;
	}
	~stack() {
		checkCudaErrors(cudaFreeHost(host));
		checkCudaErrors(cudaFree(dev));
	}
};

int factorial(int n) {
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

// Immutable combinatorial tables shared by nonlinear backward contexts.
struct derivative_tables {
    itable dTab, Crs;
    explicit derivative_tables(itable* S): dTab(S->h,S->h), Crs(S->h,S->h) {
        const int rows=S->h, variables=S->w-1;
        std::map<std::vector<int>,int> index;
        for(int r=0;r<rows;++r){
            std::vector<int> counts(variables);
            for(int v=0;v<variables;++v)counts[v]=S->el(r,v+1);
            index.emplace(std::move(counts),r);
        }
        for(int i=0;i<rows;++i)for(int j=0;j<rows;++j){
            std::vector<int> difference(variables);int coefficient=1;
            for(int v=0;v<variables;++v){
                int a=S->el(i,v+1),b=S->el(j,v+1);difference[v]=a-b;
                coefficient*=a>=b?factorial(a)/(factorial(b)*factorial(a-b)):0;
            }
            auto found=index.find(difference);
            dTab.set_el(i,j,found==index.end()?-1:found->second);
            Crs.set_el(i,j,coefficient?coefficient:-1);
        }
        dTab.todev();Crs.todev();
    }
};

class context {
public:
	int w;
	int h;
	itable* S;
	ptable* ADDR;
	ptable* ADDR_dE;
	itable* dTab;
	itable* Crs;
	stack* STCK;
    std::shared_ptr<derivative_tables> tables;

	context* next;
	context* prev;

	int matches(itable* SRC, int H, int W, int mode) {
		if (w != W || h != H) { return 0; }
		if (S->h != SRC->h || S->w != SRC->w) { return 0; }
		for (int i = 0; i < S->h*S->w; i++) {
			if (S->host[i] != SRC->host[i]) { return 0; }
		}
		if ((ADDR_dE != NULL) != (mode == 1)) return 0;
		return 1;
	}

	// mode shows if backward pass is required
	context(itable* SRC, memory* mem, int H, int W, int mode = 0, bool nonlinear = true,
            std::shared_ptr<derivative_tables> shared = nullptr) {
		w = W;
		h = H;
		S = new itable(SRC);

        ADDR = new ptable(S->h,H,W,mem);
        if(nonlinear && mode==1)tables=shared?shared:std::make_shared<derivative_tables>(S);
        dTab=tables?&tables->dTab:nullptr;
        Crs=tables?&tables->Crs:nullptr;
        STCK=nonlinear?new stack(ADDR,S,dTab,Crs):nullptr;

		ADDR_dE = NULL;

		if (mode == 1 && mem->size >= S->h*H*W*2) {
			ADDR_dE = new ptable(S->h, H, W, mem, mode);
		}
		else if(mode == 1 && mem->size < S->h*H*W*2){
			fprintf(stderr, "\tADDR_dE was not created, low mem (%d x %d x %d) vs %d \n", H, W, S->h, mem->size);
		}

		next = NULL;
		prev = NULL;

	}

	~context() {
		delete S;
		delete ADDR;
		delete ADDR_dE;
		delete STCK;
	}


};

class weights {
public:
	int Wh;
	int Ww;
	int L; // total number of matrices, 3 by default: (W, dE/dW , dE/dW)
	int act_grad; // number of active gradient matrix/vector (for RProp it's 1 or 2)
	matrix** W; // pointer to the first matrix
	matrix** T; // pointer to the first threshold vector

	matrix* act_grad_W() {
		return W[act_grad];
	}

	matrix* inact_grad_W() {
		int inact_grad;
		if (act_grad == 1) inact_grad = 2;
		else inact_grad = 1;
		return W[inact_grad];
	}

	int inact_grad(){
		int inact_grad;
		if (act_grad == 1) inact_grad = 2;
		else inact_grad = 1;
		return inact_grad;
	}

	matrix* act_grad_T() {
		return T[act_grad];
	}

	matrix* inact_grad_T() {
		int inact_grad;
		if (act_grad == 1) inact_grad = 2;
		else inact_grad = 1;
		return T[inact_grad];
	}

	matrix* get_W() {
		return W[0];
	}

	matrix* get_T() {
		return T[0];
	}

	matrix* get_delta_W() {
		return W[3];
	}

	matrix* get_delta_T() {
		return T[3];
	}

	int switch_active() {
		if (act_grad == 1) act_grad = 2;
		else act_grad = 1;
		return act_grad;
	}
	
	weights(int h, int w, int l = 4) {
		Wh = h;
		Ww = w;
		L = l;
		float* addr;
		memory* MEM;
		// weights are always on device
		checkCudaErrors(cudaHostAlloc((void**)&W, L * sizeof(matrix*), cudaHostAllocDefault));
		checkCudaErrors(cudaHostAlloc((void**)&T, L * sizeof(matrix*), cudaHostAllocDefault));
		
		for (int i = 0; i < L; i++) { 
			W[i] = new matrix(h, w + 1, 'd'); // 'w' is the number of neurons in the previous layer
			W[i]->w = w;
			addr = W[i]->mem->vals + h * w;
			MEM = new memory(addr, h, 'd');
			T[i] = new matrix(MEM, h, 1);
		//	T[i] = new matrix(h, 1, 'd');
		}


        // W[0], including its bias view, is uploaded from the Python initializer.
        for(int i=1;i<3;++i)
            checkCudaErrors(cudaMemsetAsync(W[i]->mem->vals,0,W[i]->mem->size*sizeof(float),0));
        CPDE_LAUNCH(Fill_Constant_k, 64, 64, 0, 0, W[3]->mem->size,.002f,W[3]->mem->vals);
        checkCudaErrors(cudaGetLastError());

		act_grad = 1;
	}

	~weights() {
		for (int i = 0; i < L; i++) { 
			delete T[i]; // release non-owning bias view
			delete W[i];
		}
        checkCudaErrors(cudaFreeHost(W));
        checkCudaErrors(cudaFreeHost(T));
	}
};

class layer {
public:
	int n; // number of neurons
	int lin; // 0 - sigmoid, 1 - linear.
	layer* next;
	layer* prev;
	memory* mem;
	context* env;
	weights* W_plus;
	weights* W_minus;

	layer(int N, int LIN, int W_MAX, char loc) {
		n = N;
		lin = LIN;
		next = NULL;
		prev = NULL;
		mem = new memory(W_MAX*N, loc);
		env = NULL;
		W_plus = NULL;
		W_minus = NULL;
	}


	int add_context(itable* S, int N, int W, int mode = 1,
                    std::shared_ptr<derivative_tables> shared = nullptr) {
        if (get_context(S, N, W, mode)) return 0;
        context* toadd = new context(S, mem, N, W, mode, lin==0, shared);
        context* tail = env;
        if (!tail) { env = toadd; return 0; }
        while (tail->next) tail = tail->next;
        tail->next = toadd; toadd->prev = tail;
        return 0;
    }


	context* get_context(itable* S, int H, int W, int mode = 1) {
		context* cenv = env;
		if (cenv == NULL) return NULL;
		else {
			while (cenv->prev != NULL) cenv = cenv->prev; // rewind
			if (cenv->matches(S, H, W, mode)) return cenv;
			while (cenv->next != NULL) {
				cenv = cenv->next;
				if (cenv->matches(S, H, W, mode)) return cenv;
			}
			return NULL;
		}
	}

	int wipe_context() {
        context* cur = env;
        if (cur) while (cur->prev) cur = cur->prev;
        while (cur) { context* next = cur->next; delete cur; cur = next; }
        env = NULL;
        return 0;
    }

	~layer() {
		delete mem;
		delete W_plus;
		wipe_context();
	}
};

class conf {
public:
	int layers;
	int* neurons;
	int* lin;
	int devID;
	int maxlen; // maximum number of patterns

	conf() {
		layers = 0;
		neurons = NULL;
		lin = NULL;
		devID = -1;
		maxlen = 0;
	}

	~conf() {
		free(neurons);
		free(lin);
	}

};

class inout {
public:
	int devID;
	int h;
	int w;
	memory* mem;
	itable* S;
	ptable* ADDR;

	inout(int H, int W, itable* S0, int DEV_ID = 0) {
		devID = DEV_ID;
		h = H;
		w = W;
		int devID_current;
		checkCudaErrors(cudaGetDevice(&devID_current));
		if (devID_current != devID) {
			fprintf(stderr, "\tswitching active device from %d to %d\n", devID_current, devID);
			cudaSetDevice(devID);
		}
		S = new itable(S0);
		//S->todev();
		mem = new memory(h * w * S->h, 'd');
		ADDR = new ptable(S->h, h, w, mem);

		if (devID_current != devID) cudaSetDevice(devID_current);
	}

	~inout() {
		delete mem;
		delete S;
		delete ADDR;
	}
};

class network {
public:

	cublasHandle_t handle;
	int devID;
	layer* layers;
	stream* str;
	layer* cache;
    std::map<std::vector<int>,std::weak_ptr<derivative_tables>> table_cache;
    std::shared_ptr<derivative_tables> shared_tables(itable* schema) {
        // Expired entries own no device storage; discard their host keys too.
        for(auto i=table_cache.begin();i!=table_cache.end();)
            if(i->second.expired())i=table_cache.erase(i);else ++i;
        std::vector<int> key{schema->h,schema->w};
        key.insert(key.end(),schema->host,schema->host+schema->h*schema->w);
        auto& cached=table_cache[key];auto result=cached.lock();
        if(!result){result=std::make_shared<derivative_tables>(schema);cached=result;}
        return result;
    }


	matrix* ones; // vector filled with 1.0f's

	network(conf* config) {
		int devID_current;
		int max_neurons = 0;
		if (config->maxlen > 0) {

			devID = config->devID;
			checkCudaErrors(cudaGetDevice(&devID_current));
			if (devID_current != config->devID) {
				fprintf(stderr, "\tswitching active device from %d to %d\n", devID_current, devID);
				cudaSetDevice(config->devID);
			}
			checkCudaErrors(cublasCreate(&handle));

			layer* current;
			layers = new layer(config->neurons[0], config->lin[0], config->maxlen, 'd'); // Maxlen is the maximum number of patterns for feedforward-only mode. The number of patterns for forward+backward mode is two times lower
			current = layers;
			if (max_neurons < current->n) max_neurons = current->n;
			for (int i = 1; i < config->layers; i++) {
				current->next = new layer(config->neurons[i], config->lin[i], config->maxlen, 'd');
				
				current->W_plus = new weights(config->neurons[i], config->neurons[i - 1]);
				
				current->next->W_minus = current->W_plus;
				current->next->prev = current;
				if (max_neurons < current->n) max_neurons = current->n;
				current = current->next;
			}
			max_neurons = std::max(max_neurons, current->n);
			str = new stream();
			checkCudaErrors(cublasSetStream(handle, *str->exec));

			fprintf(stderr, "\tout %d components, maxneurons: %d, maxlen: %d\n", config->neurons[config->layers - 1], max_neurons, config->maxlen);
			cache = new layer(max_neurons, 1, config->maxlen/2 + 1, 'd'); // cache never needs double capacity, neither loss (nor ones).
			ones = new matrix(1, config->maxlen/2 + 1, 'd');
			ones->init(1.0f, 'v');
			if (devID_current != config->devID) cudaSetDevice(devID_current);
		}
		else {
			fprintf(stderr, "\tincorrect conf\n");
			devID = -1;
			layers = NULL;
			str = NULL;
			cache = NULL;
			ones = NULL;
		}
	}

	int D(layer* source) {
		int order;
		context* ctx = cache->get_context(source->env->S, source->env->h, source->env->w, 0);
		for (int i = 0; i < ctx->S->h; i++) {
			order = *(ctx->S->host + i);
			if (order == 0) CPDE_LAUNCH(d0, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 1) CPDE_LAUNCH(d1, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 2) CPDE_LAUNCH(d2, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 3) CPDE_LAUNCH(d3, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 4) CPDE_LAUNCH(d4, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 5) CPDE_LAUNCH(d5, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
			if (order == 6) CPDE_LAUNCH(d6, 128, 64, 0, *str->exec, ctx->ADDR->host[i], (source->env->STCK->dev) + 64 * i, source->env->h, source->env->w);
		}
		return 0;
	}

	int prop_context() {
		layer* current = layers;
		itable* S = current->env->S;
		int width = current->env->w;
		int mode = 1;
		if (current->env->ADDR_dE == NULL) mode = 0;
		int height;
		context *fnd;
		while (current->next != NULL) {
			current = current->next;
			height = current->n;
			fnd = current->get_context(S, height, width, mode);
			if (fnd != NULL)current->env = fnd;
			else {
				fprintf(stderr, "\tunable to propagate context\n");
				return -1;
			}
		}
		return 0;
	}

	layer* feedforward() {
		layer* current = layers;
		int Ah, Bw, Bh, Ch; const float alpha = 1.0f; const float beta = 0.0f;
		while (current->next != NULL) {
			current = current->next;
			if (current->prev->lin == 0) {
				D(current->prev);
				Ah = current->W_minus->Wh;
				Bw = current->prev->env->S->h * current->prev->env->w;
				Bh = current->prev->env->h;
				Ch = Ah;
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ah, Bw, Bh, &alpha, current->W_minus->W[0]->mem->vals, Ah, cache->env->ADDR->host[0], Bh, &beta, current->env->ADDR->host[0], Ch));
				checkCudaErrors(cublasSger(handle, current->env->h, current->env->w, &alpha, current->W_minus->T[0]->mem->vals, 1, ones->mem->vals, 1, current->env->ADDR->host[0], Ch));
			}
			if (current->prev->lin == 1) {
				Ah = current->W_minus->Wh;
				Bw = current->prev->env->S->h * current->prev->env->w;
				Bh = current->prev->env->h;
				Ch = Ah;
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ah, Bw, Bh, &alpha, current->W_minus->W[0]->mem->vals, Ah, current->prev->env->ADDR->host[0], Bh, &beta, current->env->ADDR->host[0], Ch));
				checkCudaErrors(cublasSger(handle, current->env->h, current->env->w, &alpha, current->W_minus->T[0]->mem->vals, 1, ones->mem->vals, 1, current->env->ADDR->host[0], Ch));
			}
		}
		return current;
	}


	int reset_resile(float R) {
		layer* current = layers;
		while (current->prev != NULL) current = current->prev; // rewind
		int h, w;
		weights* W; 
		while (current->W_plus != NULL) {
			W = current->W_plus;
			w = W->Ww;
			h = W->Wh;
			CPDE_LAUNCH(Reset_Resile_k, 64, 64, 0, *str->exec, h, w + 1, R, W->get_delta_W()->mem->vals, W->act_grad_W()->mem->vals, W->inact_grad_W()->mem->vals);
			current = current->next;
		}

		// fin?
		std::cout << "\tResile resetted to " << R << "\n";
		return 0;
	}

	int set_resile(float value) {
		layer* current = layers;
		while (current->prev != NULL) current = current->prev;
		while (current->W_plus != NULL) {
			weights* W = current->W_plus;
			CPDE_LAUNCH(Set_Resile_k, 64, 64, 0, *str->exec, 
				W->Wh, W->Ww + 1, value, W->get_delta_W()->mem->vals,
                W->act_grad_W()->mem->vals, W->inact_grad_W()->mem->vals);
			current = current->next;
		}
		return 0;
	}

	layer* backprop(char mode = 'n', int dX = 0) { // dX requests dE/dInput
		layer* current = layers;
		while (current->next != NULL) current = current->next;
		int Ah, Aw, Bh, Bw, Ch; 
		const float alpha1 = 1.0f; 
		const float beta0 = 0.0f;

		float beta = 0.0f;
		if (mode == '+') beta = 1.0f;

		const float betaM = beta;

		while (current->prev != NULL) {
			checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, current->env->h, current->env->w, &alpha1, current->env->ADDR_dE->host[0], current->env->h, ones->mem->vals, 1, &betaM, current->W_minus->act_grad_T()->mem->vals, 1)); // dE/dThresholds

			current = current->prev;
			Ah = current->W_plus->Wh;
			Aw = current->W_plus->Ww;
			Bw = current->next->env->S->h * current->next->env->w;
			Bh = current->next->env->h;
			Ch = Aw;

			//propagating dE/dZ to previous layer
			if(current->prev!=NULL || dX == 1){
			//	if(current->prev==NULL && dX) std::cout << "\t" << Ah << " x " << Aw << " . " << Bw << " x " << Bh << " = " << " ? x" << Ch << "\n";
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Aw, Bw, Bh, &alpha1, current->W_plus->W[0]->mem->vals, Ah, current->next->env->ADDR_dE->host[0], Bh, &beta0, current->env->ADDR_dE->host[0], Ch));
				if (current->lin == 0) CPDE_LAUNCH(back_step, 256, 256, 0, *str->exec, current->env->S->dev, current->env->S->h, current->env->ADDR_dE->dev, current->env->dTab->dev, current->env->Crs->dev, current->env->STCK->dev, current->env->h, current->env->w);
			}

			Ah = current->next->n;
			Bh = current->n;
			Bw = current->env->S->h * current->env->w;
			Ch = current->next->env->h;

			// calculating dE/dW for W_plus
			if (current->lin == 0) {
				D(current);
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Ah, Bh, Bw, &alpha1, current->next->env->ADDR_dE->host[0], Ah, cache->env->ADDR->host[0], Bh, &betaM, current->W_plus->act_grad_W()->mem->vals, Ch));
			}
			if (current->lin == 1) {
				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Ah, Bh, Bw, &alpha1, current->next->env->ADDR_dE->host[0], Ah, current->env->ADDR->host[0], Bh, &betaM, current->W_plus->act_grad_W()->mem->vals, Ch));
			}
		}
		return current;
	}

	int train(int step = 0, float lrate = 0.001f) {
		layer* current = layers;
		while (current->prev != NULL) current = current->prev; // rewind
		int h, w;
		weights* W;
		while (current->next != NULL) {
			current = current->next;
			W = current->W_minus;
			w = W->Ww; // was Wh
			h = W->Wh; // was Ww
			if(step == 0){ // this means method is RPROP
			//	std::cout << "\n" << h << " " << w + 1 << "\n";
				CPDE_LAUNCH(Resile_k, 64, 64, 0, *str->exec, h, w + 1, W->get_W()->mem->vals, W->get_delta_W()->mem->vals, W->act_grad_W()->mem->vals, W->inact_grad_W()->mem->vals);
				W->switch_active();
			}
			else if(step != 0){ // if step!=0 then ADAM is engaged
				// W->act_grad_W is always just gradient, W->get_gelta_W and W->inact grad are used for first and second momentum
				CPDE_LAUNCH(ADAM_k, 64, 64, 0, *str->exec, h, w + 1, W->get_W()->mem->vals, W->act_grad_W()->mem->vals, W->get_delta_W()->mem->vals, W->inact_grad_W()->mem->vals, lrate, 0.9f, 0.999f, pow(0.9f, (float)step), pow(0.999f, (float)step));
			}
		}
		return 0;
	}

	~network() {
		layer* current = layers->next;
		int t = 0;
		delete current->prev;
		while (current->next != NULL) {
			current = current->next;
			delete current->prev;
		}
		delete current;
		delete cache;
		delete ones;
		delete str;
		checkCudaErrors(cublasDestroy(handle));
	}
};


