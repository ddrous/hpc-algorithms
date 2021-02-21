#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include <iostream>
#include <omp.h>
#include <cassert>
#include <vector>

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

void printAvailableCores(){
    const std::vector<long int> cores = GetBindingList();

    std::cout << "Available cores = ";
    
    for(long int core : cores){
        std::cout << core << "  ";
    }
}


int main(){
    std::cout << "Start parallel region" << std::endl;
    
    // TODO parallel section
    #pragma omp parallel
    {
        // TODO one thread at a time
        #pragma omp critical
        {
            // std::cout << "I am thread " << /*omp thread id*/ -1 << "/" << /*omp nb threads*/ -1 << " ";
            std::cout << "I am thread " << omp_get_thread_num() << "/" << omp_get_num_threads() << " ";
            printAvailableCores();
            std::cout << std::endl;            
        }

        // TODO all threads wait each other
        #pragma omp barrier
        
        // TODO only master
        #pragma omp master
        std::cout << "I am master, all threads done " << omp_get_thread_num() << std::endl;
                
        for(int idxThread = 0 ; idxThread < omp_get_num_threads() ; ++idxThread){
            if(idxThread == omp_get_thread_num()){      // PRINT CORES IN ORDER !
                std::cout << "I am thread " << omp_get_thread_num() << std::endl;
            }
            // TODO all threads wait each other
            #pragma omp barrier
        }
    }
    return 0;
}