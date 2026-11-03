"""Independent NumPy forward reference, including raw spatial derivatives.

Uses explicit chain rules through order three. Complex weights are supported
for complex-step directional checks of parameter gradients, without CUDA or
the expression compiler taking part in the reference differentiation.
"""
import numpy as np


def forward(weights, layers, linear, points, order):
    dtype=np.result_type(*[w.dtype for w in weights],np.float64)
    h=np.asarray(points,dtype=dtype)
    n,d=h.shape
    first=np.broadcast_to(np.eye(d,dtype=dtype)[:,None,:],(d,n,d)).copy()
    second=np.zeros((d,d,n,d),dtype=dtype)
    third=np.zeros((d,d,d,n,d),dtype=dtype) if order==3 else None
    for layer,(width,height) in enumerate(zip(layers[:-1],layers[1:])):
        matrix=weights[layer][:width*height].reshape((height,width),order='F')
        bias=weights[layer][width*height:]
        z=h@matrix.T+bias
        z1=first@matrix.T
        z2=second@matrix.T if order>=2 else None
        z3=third@matrix.T if order==3 else None
        if linear[layer+1]:
            h,first=z,z1
            if order>=2: second=z2
            if order==3: third=z3
            continue
        h=1/(1+np.exp(-z))
        s1=h*(1-h); s2=s1*(1-2*h); s3=s1*(1-6*s1)
        first=s1*z1
        if order>=2:
            second=np.empty_like(z2)
            for i in range(d):
                for j in range(d): second[i,j]=s1*z2[i,j]+s2*z1[i]*z1[j]
        if order==3:
            third=np.empty_like(z3)
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        third[i,j,k]=s1*z3[i,j,k]+s2*(z2[i,j]*z1[k]+z2[i,k]*z1[j]+z2[j,k]*z1[i])+s3*z1[i]*z1[j]*z1[k]
    return h,first,second,third


def rows(jet,schema):
    result=[]
    for row in schema:
        axes=tuple(axis for axis,count in enumerate(row[1:]) for _ in range(int(count)))
        result.append(jet[int(row[0])][axes] if axes else jet[0])
    return np.stack(result).transpose(0,2,1)
