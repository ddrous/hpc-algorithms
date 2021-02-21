#ifndef _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
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

void BindToCore(const int inCoreId){
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(inCoreId, &set);

    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    int retValue = sched_setaffinity(tid, sizeof(set), &set);
    assert(retValue == 0);
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


double dot(const double* vec0, const double* vec1, const long int N){
   double sum = 0;
   for(long int idx = 0 ; idx < N ; ++idx){
       sum += vec0[idx] * vec1[idx];
   }
   return sum;
}

void test(){
    const std::vector<long int> availableCores = GetBindingList();
    BindToCore(int(availableCores[0]));

    const long int TestSize = 500000;
    const long int NbLoops = 1000;

    std::cout << "Check dot" << std::endl;
    std::cout << "TestSize = " << TestSize << std::endl;

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(0, 1);

    std::vector<double> vec0(TestSize);
    std::vector<double> vec1(TestSize);

    double currentSum = 0;
    for(long int idxVal = 0 ; idxVal < TestSize ; ++idxVal){
        vec0[idxVal] = dis(gen);
        vec1[idxVal] = dis(gen);
        currentSum += vec0[idxVal]*vec1[idxVal];
    }
    currentSum *= NbLoops;

    {
        double scalarSum = 0;
        Timer timerScalar;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarSum += dot(vec0.data(), vec1.data(), TestSize);
        }
        timerScalar.stop();

        std::cout << ">> Without move : " << timerScalar.getElapsed() << std::endl;

        CheckEqual(currentSum,scalarSum);
    }
    {
        size_t idxCore = 0;
        double scalarWithStopSum = 0;
        Timer timerWithStop;

        cpu_set_t  mask;        // Createa an affinity mask

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarWithStopSum += dot(vec0.data(), vec1.data(), TestSize);
            // TODO BindToCore to correct core

            //------------------------------------------------------------------
            CPU_ZERO(&mask);    // Empty out the mask
            size_t newCPUindex = size_t(idxLoop) % availableCores.size();       // The new cpu core's index
            idxCore = availableCores[newCPUindex];                              // The new cpu core
            CPU_SET(idxCore, &mask);                                            // Add only the new core to the mask
            int result = sched_setaffinity(0, sizeof(mask), &mask);             // Ask the calling process to use this mask
            
            /* For testing purposes */
            // if (result == -1)
            //     exit(1);
            // int current_cpu = sched_getcpu();
            // std::cout << "curent cpu core : " << current_cpu << std::endl;
            //------------------------------------------------------------------

        }
        timerWithStop.stop();

        std::cout << ">> With move : " << timerWithStop.getElapsed() << std::endl;

        CheckEqual(currentSum,scalarWithStopSum);
    }
}

int main(){
    test();

    return 0;
}