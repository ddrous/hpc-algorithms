/*
long int dot(const long int* vec0, const long int* vec1, const long int N){
   long int sum = 0;
   for(long int idx = 0 ; idx < N ; ++idx){
       sum += vec0[idx] * vec1[idx];
   }
   return sum;
}
*/

// extern "C" long int dot(const long int* vec0, const long int* vec1,
//                              const long int N);
// __asm__(
// ".global dot \n"
// "dot: \n"
// "ret;\n"
// );

extern "C" long int dot(const long int* vec0, const long int* vec1,
                             const long int N);
__asm__(
".global dot \n"
"dot: \n"
        "movq    %rdx, %rcx; \n"
        "testq   %rdx, %rdx; \n"
        "jle     .L4 \n"
        "xorl    %eax, %eax; \n"
        "xorl    %r8d, %r8d; \n"
".L3: \n"
        "movq    (%rdi,%rax,8), %rdx; \n"
        "imulq   (%rsi,%rax,8), %rdx; \n"
        "addq    $1, %rax; \n"
        "addq    %rdx, %r8; \n"
        "cmpq    %rax, %rcx; \n"
        "jne     .L3 \n"
        "movq    %r8, %rax; \n"
        "ret \n"
".L4: \n"
        "xorl    %r8d, %r8d; \n"
        "movq    %r8, %rax; \n"
        "ret; \n"
);


#include "utils.hpp"
#include <iostream>
#include <random>
#include <vector>

void test(){
    const long int TestSize = 1000;

    std::cout << "Check dot" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long int> dis(1, 10);

    for(long int idx = 0 ; idx < TestSize ; ++idx){
        std::vector<long int> vec0(idx);
        std::vector<long int> vec1(idx);

        long int currentSum = 0;
        for(long int idxVal = 0 ; idxVal < idx ; ++idxVal){
            vec0[idxVal] = dis(gen);
            vec1[idxVal] = dis(gen);
            currentSum += vec0[idxVal]*vec1[idxVal];
        }

        CheckEqual(currentSum, dot(vec0.data(), vec1.data(), idx));
    }
}

int main(){
    test();

    return 0;
}
