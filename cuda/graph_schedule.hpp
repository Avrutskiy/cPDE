#pragma once
#include <vector>
#include "scalar_ops.cuh"

namespace cpde_graph {
// Determine which leaf multipliers are known before evaluating any integrals.
// Unknown values stay unknown; this is dependency analysis, not a symbolic
// simplifier. Attach has value a but sends its adjoint to both a and b.
struct KnownScalar { bool known; double value; };
inline KnownScalar known(double v){return {std::isfinite(v),v};}
inline KnownScalar sum(KnownScalar a,KnownScalar b){
    return a.known&&b.known?known(a.value+b.value):KnownScalar{false,0};
}
inline KnownScalar product(KnownScalar a,KnownScalar b){
    if((a.known&&a.value==0)||(b.known&&b.value==0))return known(0);
    return a.known&&b.known?known(a.value*b.value):KnownScalar{false,0};
}
inline std::vector<KnownScalar> schedule(const std::vector<cpde_scalar::Node>& nodes,int root,int leaves){
    using namespace cpde_scalar;
    const KnownScalar unknown{false,0};
    std::vector<KnownScalar> values(nodes.size(),unknown),da(nodes.size(),unknown),db(nodes.size(),unknown);
    std::vector<bool> active(nodes.size(),false);
    for(size_t i=0;i<nodes.size();++i){
        auto n=nodes[i];
        if(n.op==Constant){values[i]=known(n.parameter);continue;}
        if(n.op==Leaf){active[i]=true;continue;}
        bool binary=n.op<=Div||n.op==Attach;
        auto a=values[n.a],b=binary?values[n.b]:known(0);
        active[i]=active[n.a]||(binary&&active[n.b]);
        if(a.known&&(b.known||n.op==Attach))values[i]=known(value(n.op,a.value,b.value,n.parameter));
        if(n.op==Add||n.op==Attach){da[i]=db[i]=known(1);}
        else if(n.op==Sub){da[i]=known(1);db[i]=known(-1);}
        else if(n.op==Mul){da[i]=b;db[i]=a;}
        else if(n.op==Div){
            if(b.known&&b.value!=0){da[i]=known(1/b.value);if(a.known)db[i]=known(-(a.value/b.value)/b.value);}
        }
        else if(n.op==Neg)da[i]=known(-1);
        else if(n.op==cpde_scalar::Power&&(n.parameter==0||n.parameter==1))da[i]=known(n.parameter);
        else if(a.known&&values[i].known){
            double x,y;partials(n.op,a.value,b.value,values[i].value,n.parameter,x,y);da[i]=known(x);
        }
    }
    std::vector<KnownScalar> adjoints(nodes.size(),known(0)),multipliers(leaves,known(0));
    adjoints[root]=known(1);
    for(int i=int(nodes.size())-1;i>=0;--i){
        auto n=nodes[i];auto seed=adjoints[i];
        if(!active[i]||(seed.known&&seed.value==0))continue;
        if(n.op==Leaf)multipliers[n.a]=sum(multipliers[n.a],seed);
        else {
            if(active[n.a])adjoints[n.a]=sum(adjoints[n.a],product(seed,da[i]));
            if((n.op<=Div||n.op==Attach)&&active[n.b])adjoints[n.b]=sum(adjoints[n.b],product(seed,db[i]));
        }
    }
    return multipliers;
}
}
