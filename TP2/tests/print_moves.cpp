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

#include <unistd.h>     // For the process to sleep
#include <vector>
#include <map>


//--------------------------------------------------------

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
//--------------------------------------------------------



int main(){
// TODO

//------------------------------------------------------------------------------
    const std::vector<long int> coreList = GetBindingList();    // all the avaliable cores
    std::map<long int, int> coreCount;                          // counts the appearances of each core
    for (long int core: coreList){
        coreCount[core] = 0;
    }
    
    int first_loop = true;
    int previous_cpu, current_cpu;              // Hold the previous and current cpu affinity for the process
    while(true){ 
        current_cpu = sched_getcpu();           // Get the current affinity

        if (first_loop == true){                // In the first iteration, the previous core is the current one 
            previous_cpu = current_cpu;
            coreCount[current_cpu] += 1;
            first_loop = false;
        }

        if(current_cpu != previous_cpu){        // Check if the process has changed cores
            coreCount[current_cpu] += 1;
            std::cout << "\nMoved to core " << current_cpu << std::endl;
            
            std::cout << "Core count: " ;
            for(const auto &core : coreCount)
                std::cout << "[" << core.first << "] " << core.second << "   ";
            
            previous_cpu = current_cpu;
        }

        /* For testing purposes */
        // sleep(2);                                                           // Sleep for 2 seconds
        // std::cout << "\nCurrent CPU core: "<< current_cpu << std::endl;     // For testting purposes
    }
//------------------------------------------------------------------------------
    return 0;
}