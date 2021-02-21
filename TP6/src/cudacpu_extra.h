#ifndef CUDACPU_EXTRA_H
#define CUDACPU_EXTRA_H

#include "cudacpu_struct.h"

int dim3_sum(const dim3& inPos){
    return inPos.x + inPos.y + inPos.z;
}

int dim3_mul(const dim3& inPos){
    return inPos.x * inPos.y * inPos.z;
}

int3 dim3_1DidxToPos(const int inGlobal1Didx, const dim3 inSize){
    int3 idx;
    idx.x = inGlobal1Didx / (inSize.z*inSize.y);
    idx.y = (inGlobal1Didx % (inSize.z*inSize.y))/inSize.z;
    idx.z = inGlobal1Didx % inSize.z;
    return idx;
}

#endif
