#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <cassert>
#include <omp.h>


int Fibonacci(int n) {
  if (n < 2)
    return n;
  else
    return Fibonacci(n-1) + Fibonacci(n-2);
}


// /* FIRST VERSION WITHOUT OPTIMIZATIONS */
// int FibonacciOmp(int n) {
//     // TODO implement Fibonnaci with tasks

//     if (n < 2)
//         return n;

//     else {

//     int val1, val2;

//         #pragma omp parallel // create the threads
//         {
//             // here there might be 10 threads
//             #pragma omp master // create a section for the master thread only
//             // the other threads will continue
//             {
//                 // Create a task that can be executed by any threads
//                 #pragma omp task
//                 val1 = FibonacciOmp(n-1);

//                 // Create a task that can be executed by any threads
//                 #pragma omp task
//                 val2 = FibonacciOmp(n-2);

//                 // The master thread will do this
//                 // // do nothing !!!

//                 // The master threads waits for the two task to finish
//                 #pragma omp taskwait
//             }
//         } // The other threads are waiting here

//         return val1 + val2;

//     }

// }


/* SECOND VERSION WITH OPTIMIZATIONS */
int FibonacciOmp(int n) {
    // TODO implement Fibonnaci with tasks

    if (n < 2)
        return n;

    else if (n < 15) {
        return FibonacciOmp(n-1) + FibonacciOmp(n-2);
    }

    else {
        int val1, val2, res;

        // Create a task that does this
        #pragma omp task shared(val1) firstprivate(n) priority(2)
        val1 = FibonacciOmp(n-1);

        // The master thread does this
        val2 = FibonacciOmp(n-2);

        // The master threads waits for the task to finish
        #pragma omp taskwait
        res = val1 + val2;

        return res;

    }

}


/* THIRD VERSION WITH DYNAMIC PROGRAMMING */
int FibonacciOmpDy(int n, std::map<int,int>& fiboMap) {

    if (fiboMap.find(n) != fiboMap.end())
        return fiboMap[n];

    else {

        if (n < 2)
            fiboMap[n] = n;

        else if (n < 20) {
            fiboMap[n] = FibonacciOmpDy(n-1, fiboMap) + FibonacciOmpDy(n-2, fiboMap);
        }

        else {
            int val1, val2;
                // Create a task 
                #pragma omp task shared(val1) firstprivate(n)
                val1 = FibonacciOmpDy(n-1, fiboMap);

                // The master does this
                val2 = FibonacciOmpDy(n-2, fiboMap);

                // The master threads waits
                #pragma omp taskwait
                fiboMap[n] = val1 + val2;
        }

        return fiboMap[n];
    }

}


void test(){
    const long int TestSize = 40;
    const long int NbLoops = 10;

    std::cout << "Check Fibonacci" << std::endl;
    std::cout << "TestSize = " << TestSize << std::endl;

    int scalarFibonnaci = 0;
    {
        Timer timerSequential;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarFibonnaci += Fibonacci(TestSize);
        }
        timerSequential.stop();

        std::cout << ">> Sequential timer : " << timerSequential.getElapsed() << std::endl;
    }
    #pragma omp parallel
    #pragma omp master
    {

        int ompFibonnaci = 0;
        Timer timerParallel;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            ompFibonnaci += FibonacciOmp(TestSize);
        }

        timerParallel.stop();

        std::cout << ">> There are " << omp_get_num_threads() << " threads" << std::endl;
        std::cout << ">> Omp timer : " << timerParallel.getElapsed() << std::endl;

        int ompFibonnaciDy = 0;
        std::map<int,int> fiboMap;
        timerParallel.reset();
        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            ompFibonnaciDy += FibonacciOmpDy(TestSize, fiboMap);
        }
        timerParallel.stop();

        std::cout << ">> Omp timer dynamic : " << timerParallel.getElapsed() << std::endl;

        CheckEqual(scalarFibonnaci,ompFibonnaci);
        CheckEqual(scalarFibonnaci,ompFibonnaciDy);
    }
}

int main(){
    test();

    return 0;
}