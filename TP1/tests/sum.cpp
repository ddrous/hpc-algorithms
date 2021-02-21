
/*
long int sum2(const long int val0, const long int val1){
   return val0+val1;
}
*/
// extern "C" long int sum2_asm(const long int val0, const long int val1);
// __asm__(
// ".global sum2_asm \n"
// "sum2_asm: \n"
// "ret;\n"
// );
extern "C" long int sum2_asm(const long int val0, const long int val1);
__asm__(
".global sum2_asm \n"
"sum2_asm: \n"
"leaq   (%rdi,%rsi), %rax \n"
"ret; \n"
);


/*
long int sum7(const long int val0, const long int val1,
                             const long int val2, const long int val3,
                             const long int val4, const long int val5,
                             const long int val6){
   return val0+val1+val2+val3+val4+val5+val6;
}
*/
// extern "C" long int sum7_asm(const long int val0, const long int val1,
//                              const long int val2, const long int val3,
//                              const long int val4, const long int val5,
//                              const long int val6);
// __asm__(
// ".global sum7_asm \n"
// "sum7_asm: \n"
// "ret;\n"
// );
extern "C" long int sum7_asm(const long int val0, const long int val1,
                             const long int val2, const long int val3,
                             const long int val4, const long int val5,
                             const long int val6);
__asm__(
".global sum7_asm \n"
"sum7_asm: \n"
"addq    %rsi, %rdi \n"
"addq    %rdx, %rdi \n"
"addq    %rcx, %rdi \n"
"addq    %r8, %rdi \n"
"leaq    (%rdi,%r9), %rax \n"
"addq    8(%rsp), %rax \n"
"ret;\n"
);


#include "utils.hpp"
#include <iostream>
#include <random>
#include <array>

void test(){
    const long int TestSize = 1000;

    std::cout << "Check sum2_asm" << std::endl;

    for(long int idx = 0 ; idx < TestSize ; ++idx){
        CheckEqual(idx+idx-1,sum2_asm(idx, idx-1));
        CheckEqual(idx+idx+1,sum2_asm(idx, idx+1));
        CheckEqual(idx+idx*2,sum2_asm(idx, idx*2));
    }

    std::cout << "Check sum7_asm" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long int> dis(1, 1000000);

    for(long int idx = 0 ; idx < TestSize ; ++idx){
        std::array<long int, 7> vals;
        long int currentSum = 0;
        for(long int idxVal = 0 ; idxVal < 7 ; ++idxVal){
            vals[idxVal] = dis(gen);
            currentSum += vals[idxVal];
        }

        CheckEqual(currentSum,sum7_asm(vals[0], vals[1], vals[2], vals[3],
                                       vals[4], vals[5], vals[6]));
    }
}

int main(){
    test();

    return 0;
}