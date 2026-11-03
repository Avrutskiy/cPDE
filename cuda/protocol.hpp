#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>

// Local little-endian protocol. Frame: uint64 byte length, uint32 kind, payload.
namespace wire {
struct Buffer {
    std::vector<unsigned char> bytes;
    size_t pos=0;
    template<class T> T get() {
        if(pos+sizeof(T)>bytes.size()) throw std::runtime_error("truncated command");
        T x; std::memcpy(&x,bytes.data()+pos,sizeof(T));pos+=sizeof(T);return x;
    }
    template<class T> void put(T x) {
        auto p=reinterpret_cast<const unsigned char*>(&x);bytes.insert(bytes.end(),p,p+sizeof(T));
    }
    template<class T> std::vector<T> array() {
        uint64_t n=get<uint64_t>();
        if(n>(bytes.size()-pos)/sizeof(T)) throw std::runtime_error("truncated array");
        std::vector<T> v(n); if(n)std::memcpy(v.data(),bytes.data()+pos,n*sizeof(T));
        pos+=n*sizeof(T);return v;
    }
    template<class T> void array(const std::vector<T>& a) {
        put<uint64_t>(a.size());if(a.empty())return;
        auto p=reinterpret_cast<const unsigned char*>(a.data());bytes.insert(bytes.end(),p,p+a.size()*sizeof(T));
    }
    std::string string(){auto a=array<char>();return std::string(a.begin(),a.end());}
    void string(const std::string& s){array(std::vector<char>(s.begin(),s.end()));}
    void end(){if(pos!=bytes.size())throw std::runtime_error("unexpected command fields");}
};
inline bool read(Buffer& b) {
    uint64_t n;
    if(std::fread(&n,1,sizeof(n),stdin)!=sizeof(n))return false;
    if(n<4 || n>uint64_t(4)*1024*1024*1024)throw std::runtime_error("invalid frame size");
    b.bytes.resize(n);
    size_t got=0;
    while(got<n){size_t k=std::fread(b.bytes.data()+got,1,n-got,stdin);if(!k)throw std::runtime_error("truncated frame");got+=k;}
    return true;
}
inline void send(uint32_t kind,const Buffer& b=Buffer()) {
    uint64_t n=4+b.bytes.size();
    if(std::fwrite(&n,1,8,stdout)!=8 || std::fwrite(&kind,1,4,stdout)!=4 ||
       (!b.bytes.empty() && std::fwrite(b.bytes.data(),1,b.bytes.size(),stdout)!=b.bytes.size()))
        throw std::runtime_error("response pipe closed");
    std::fflush(stdout);
}
}
