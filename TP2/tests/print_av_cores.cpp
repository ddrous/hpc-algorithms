#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include <iostream>
#include <vector>
#include <cassert>

long int GetBinding(){
    cpu_set_t mask;
    CPU_ZERO(&mask);
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    // Get the affinity
    int retValue = sched_getaffinity(tid, sizeof(mask), &mask);
    assert(retValue == 0);
    long int retMask = 0;
    for(size_t idx = 0 ; idx < sizeof(long int)*8-1 ; ++idx){
        if(CPU_ISSET(idx, &mask)){
            retMask |= (1<<idx);
        }
    }
    return retMask;
}


std::vector<long int> GetBindingList(){
    const long int cores = GetBinding();
    
    std::vector<long int> list;
    long int idx = 0;
    while((1 << idx) <= cores){
        if((1 << idx) & cores){
             list.push_back(idx);
        }
        idx += 1;
    }
    
    return list;
}


int main(){
    const std::vector<long int> cores = GetBindingList();

    std::cout << "Available cores = " << std::endl;
    
    for(long int core : cores){
        std::cout << core << "  ";
    }
    
    std::cout << std::endl;

    return 0;
}