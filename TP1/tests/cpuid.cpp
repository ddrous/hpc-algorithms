
enum RegistersNum {
    EaxRegister = 0,
    EbxRegister,
    EcxRegister,
    EdxRegister
};

bool CPUInfoGetEAX(const unsigned int CPUInfo[4], const int position){
    return (CPUInfo[EaxRegister] & (1 << position)) != 0;
}

bool CPUInfoGetEBX(const unsigned int CPUInfo[4], const int position){
    return (CPUInfo[EbxRegister] & (1 << position)) != 0;
}

bool CPUInfoGetECX(const unsigned int CPUInfo[4], const int position){
    return (CPUInfo[EcxRegister] & (1 << position)) != 0;
}

bool CPUInfoGetEDX(const unsigned int CPUInfo[4], const int position){
    return (CPUInfo[EdxRegister] & (1 << position)) != 0;
}

/*
// Else we have to ask the CPU directly by executing cpuid.
// eax should contain the information querry argument.
// Then we have to take the results from the different registers.
//
//    From : http://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html
//
//    asm ( assembler template
//        : output operands                  // optional
//        : input operands                   // optional
//        : list of clobbered registers      // optional
//        );
//
//    +---+--------------------+
//    | r |    Register(s)     |
//    +---+--------------------+
//    | a |   %eax, %ax, %al   |
//    | b |   %ebx, %bx, %bl   |
//    | c |   %ecx, %cx, %cl   |
//    | d |   %edx, %dx, %dl   |
//    | S |   %esi, %si        |
//    | D |   %edi, %di        |
//    +---+--------------------+
//

void cpuid(unsigned int CPUInfo[4],unsigned int InfoTypeEax, unsigned int InfoTypeEcx){
    __asm__ __volatile__ (
        "cpuid":            // Execute this instruction
        "=a" (CPUInfo[EaxRegister]),  // Store eax in 0
        "=b" (CPUInfo[EbxRegister]),  // Store ebx in 1
        "=c" (CPUInfo[EcxRegister]),  // Store ecx in 2
        "=d" (CPUInfo[EdxRegister]) : // Store edx in 3
        "a" (InfoTypeEax),      // Input InfoType in eax before instruction
        "c" (InfoTypeEcx)
    );
}

*/

extern "C" long int cpuid(unsigned int CPUInfo[4], unsigned int InfoTypeEax, unsigned int InfoTypeEcx);
__asm__(
".global cpuid \n"
"cpuid: \n"
// Save rbx, move %esi to %eax, and %edx to %ecx
"push %rbx; \n"
"movl %esi, %eax \n"
"movl %edx, %ecx \n"
"cpuid;\n"
// Save %eax, %ebx, %ecx, %edx into CPUInfo, and restore rbx
"movl %eax, (%rdi) \n"
"movl %ebx, 4(%rdi) \n"
"movl %ecx, 8(%rdi) \n"
"movl %edx, 12(%rdi) \n"
"pop %rbx; \n"
"ret;\n"
);


#include "utils.hpp"
#include <iostream>
#include <cassert>

void test(){
    unsigned int info[4];
    cpuid(info,0x00000001U, 0);

    std::cout << "Has HTT:" << CPUInfoGetEDX(info, 28) << std::endl;
    std::cout << "Has MMX:" << CPUInfoGetEDX(info, 23) << std::endl;
    std::cout << "Has SSE:" << CPUInfoGetEDX(info, 25) << std::endl;
    std::cout << "Has SSE2:" << CPUInfoGetEDX(info, 26) << std::endl;
    assert(CPUInfoGetEDX(info, 26));
}

int main(){
    test();

    return 0;
}